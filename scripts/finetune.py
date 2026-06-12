"""
使用原始音频文件 + metadata.jsonl 进行微调
- 使用原始音频可以方便地对连续语音进行 pack，增强模型处理长音频的能力
- 按 start_time 排序，get_item 拿到 idx 后，以概率 packing_p（默认 0.25） 判断是否 pack 下一条，
  后续继续以 packing_p 判断是否继续 pack，音频总长度不超过 packing_max_s（默认 30s）
"""

import argparse
import os

import evaluate
import torch
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="finetune",
        description="fine-tune whisper model with raw audio packing",
    )
    parser.add_argument("--name", default="ft")
    parser.add_argument("--output_dir", default="./outputs/finetune")

    # model
    parser.add_argument("--model_name", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--attn_implementation", default="sdpa")

    # dataset
    parser.add_argument("--json_data_file", default="./outputs/data/metadata.jsonl")
    parser.add_argument("--audio_path", default="./outputs/data/audio.wav")
    parser.add_argument("--valid_size", type=float, default=0)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--packing_p", type=float, default=0.25)
    parser.add_argument("--packing_max_s", type=float, default=30.0)

    # train
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=float, default=0.1)
    parser.add_argument("--save_steps", type=float, default=0.2)
    parser.add_argument("--eval_steps", type=float, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_accumulation_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to_wandb", action="store_true")

    return parser.parse_args()


def main() -> None:
    from scripts.utils import get_utc_time_str
    from src.data_manager import DataManager

    args = parse_args()
    if args.eval_steps > 0 and args.valid_size <= 0:
        raise ValueError("--eval_steps > 0 requires --valid_size > 0.")

    experiment_time_str = get_utc_time_str()
    output_dir = os.path.join(args.output_dir, args.name)
    os.environ["WANDB_DIR"] = os.path.abspath(output_dir)
    dtype = torch.bfloat16

    # ---------------- model ----------------
    processor = WhisperProcessor.from_pretrained(args.model_name, local_files_only=True)
    model_config = WhisperConfig.from_pretrained(args.model_name, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        config=model_config,
        dtype=dtype,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    )

    # freeze encoder
    model.model.encoder.requires_grad_(False)

    # ---------------- dataset ----------------
    dataset = load_dataset(
        "json",
        data_files=args.json_data_file,
        split="train",
    )
    dataset = dataset.sort("start_time")
    train_metadata = dataset
    eval_metadata = None
    if args.valid_size > 0:
        # packing 会包含片段之间的原始音频；必须按时间连续切分，避免音频和 label 对不齐。
        split_dataset = dataset.train_test_split(test_size=args.valid_size, shuffle=False)
        train_metadata = split_dataset["train"]
        eval_metadata = split_dataset["test"]

    train_dm = DataManager(
        dataset=train_metadata,
        audio_path=args.audio_path,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        packing_p=args.packing_p,
        packing_max_s=args.packing_max_s,
    )
    train_dataset = train_dm.get_dataset(num_proc=args.num_proc)

    eval_dataset = None
    compute_metrics = None
    if eval_metadata is not None and args.eval_steps > 0:
        eval_dm = DataManager(
            dataset=eval_metadata,
            audio_path=args.audio_path,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            packing_p=0.0,
            packing_max_s=args.packing_max_s,
        )
        eval_dataset = eval_dm.get_dataset(num_proc=args.num_proc)

        metric_cer = evaluate.load("cer")

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            # replace -100 with the pad_token_id
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

            # we do not want to group tokens when computing the metrics
            pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)  # type: ignore

            return {"cer": cer}

    # ---------------- train ----------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        bf16=True,
        tf32=True,
        torch_compile=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        predict_with_generate=True,
        generation_max_length=model_config.max_target_positions,
        eval_strategy="steps" if eval_dataset is not None else "no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,  # 将 eval 的中间结果移动到内存中，防止显存溢出
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        # load_best_model_at_end=True,
        # metric_for_best_model="cer",
        # greater_is_better=False,
        seed=args.seed,
        report_to="wandb" if args.report_to_wandb else "none",
        run_name=f"{args.name}_{experiment_time_str}",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=train_dm.get_collator(dtype=dtype),
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    trainer.train()


if __name__ == "__main__":
    main()
