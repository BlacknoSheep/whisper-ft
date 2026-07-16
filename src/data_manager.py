from dataclasses import dataclass

from typing import Any, Dict, List

import regex
import torch
from datasets import Audio, Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from .utils import get_prefix_prompt, normalize_text

NON_TEXT_RE = regex.compile(r"[^\p{L}\p{N}]+")
WHITESPACE_RE = regex.compile(r"\s+")


@dataclass
class SimpleCollator:
    """
    pad input_features 和 labels，返回的 labels 中 padding 部分被替换成了 -100，以正确计算 loss
    whisper forward 时会自动根据 labels 生成 decoder_input_ids（右移并添加 decoder_start_token_id ，将 -100 替换为 pad_token_id）
    由于是 causal attention，且 loss 会忽略 -100，所以不需要 decoder_attention_mask
    """

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer
    dtype: torch.dtype

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # [B, 128, 3000]，whisper 固定 pad 到 30s，所以不需要手动 pad
        input_features = torch.tensor([feature["input_features"] for feature in features], dtype=self.dtype)
        batch = self.tokenizer.pad(
            {"input_ids": [feature["labels"] for feature in features]},
            return_tensors="pt",
            return_attention_mask=True,
        )
        # replace padding with -100 to ignore loss correctly
        labels = batch.input_ids.masked_fill(batch.attention_mask.ne(1), -100)

        return {"input_features": input_features, "labels": labels}


class SimpleDataManager:
    def __init__(
        self,
        dataset: Dataset,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        samplerate=16000,
    ) -> None:
        """
        dataset 必须包含这些列：['audio', 'language', 'transcription']
        """
        assert "audio" in dataset.column_names, "Dataset must contain 'audio' column."
        assert "language" in dataset.column_names, "Dataset must contain 'language' column."
        assert "transcription" in dataset.column_names, "Dataset must contain 'transcription' column."

        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.samplerate = samplerate

    def _normalize(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入，不 batch
        """
        # 会被自动 pad 到 30s，所以不需要手动 pad
        example["input_features"] = self.feature_extractor(
            example["audio"]["array"], sampling_rate=self.samplerate
        ).input_features[0]
        label_texts = get_prefix_prompt(
            language=example["language"],
            task="transcribe",
            predict_timestamps=False,
        )
        label_texts += normalize_text(example["transcription"])
        label_texts += self.tokenizer.eos_token  # type: ignore
        example["labels"] = self.tokenizer.encode(label_texts, add_special_tokens=False)

        sot_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if example["labels"][0] == sot_id:
            example["labels"] = example["labels"][1:]  # 去掉开头的 <|startoftranscript|>

        return example

    def get_dataset(self) -> Dataset:
        """
        返回的数据集包含以下列：["input_featrues", "labels"]
        """
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=self.samplerate))
        # 由于需要根据 language 构造，不 batch
        # 由于用到了 tokenizer，所以不能多线程并行, num_proc=0
        self.dataset = self.dataset.map(
            self._normalize, remove_columns=self.dataset.column_names, load_from_cache_file=False
        )
        return self.dataset

    def get_collator(self, dtype: torch.dtype = torch.float32):
        return SimpleCollator(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            dtype=dtype,
        )
