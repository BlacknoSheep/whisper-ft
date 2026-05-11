from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset
import os
import evaluate
import torch

from src.data_manager import SimpleDataManager

output_dir = "./outputs/whisper"
num_proc = 8
device = torch.device("cuda")
dtype = torch.bfloat16
os.environ.setdefault("WANDB_LOG_MODEL", "./outputs/log")
model_name = "openai/whisper-large-v3-turbo"
max_length = 448 # model.max_target_positions
metric_cer = evaluate.load("cer")

processor = WhisperProcessor.from_pretrained(model_name)


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


dataset = load_dataset(
    "json",
    data_files="./downloads/my-dataset/metadata.jsonl",
    split="train",
)

dataset = dataset.map(
    lambda x: {"audio": os.path.join("./downloads/my-dataset", x["audio"])},
    num_proc=num_proc,
)

dm = SimpleDataManager(
    dataset=dataset,
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
)

dataset = dm.get_dataset(num_proc=num_proc)

train_valid_dataset = dataset.train_test_split(0.1, seed=42)

model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    dtype=dtype,
    # attn_implementation="flash_attention_4", # https://github.com/Dao-AILab/flash-attention/issues/2440#issuecomment-4417289210
)

model.model.encoder.requires_grad_(False)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=32,
    # eval_accumulation_steps=1,  # 将 eval 的中间结果移动到内存中，防止显存溢出
    bf16=True,  # 动态范围比 fp16 大
    tf32=True,  # gpu 加速
    max_steps=100,
    predict_with_generate=True,
    generation_max_length=max_length,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=2e-5,
    warmup_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    # metric_for_best_model="cer",
    # greater_is_better=False,
    # remove_unused_columns=False,  # keep all columns to compute loss
    seed=42,
    # report_to="wandb"
)

trainer = Seq2SeqTrainer(
    model=model,
    data_collator=dm.get_collator(dtype=dtype),
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=train_valid_dataset["train"],
    eval_dataset=train_valid_dataset["test"],  # type: ignore
    processing_class=processor,
)

trainer.train()
