import torch
from datasets import Dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
import unicodedata
import regex


@dataclass
class SimpleCollator:
    """
    pad input_features 和 labels，返回的 labels 中 padding 部分被替换成了 -100，以正确计算 loss
    """

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer
    dtype: torch.dtype

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # [B, 128, 3000]
        input_features = torch.tensor(
            [feature["input_features"] for feature in features], dtype=self.dtype
        )

        batch = self.tokenizer.pad(
            {"input_ids": [feature["labels"] for feature in features]},
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = batch.input_ids.masked_fill(batch.attention_mask.ne(1), -100)
        # 删掉 bos token，因为后续会自动加上，避免重复
        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

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
        assert "language" in dataset.column_names, (
            "Dataset must contain 'language' column."
        )
        assert "transcription" in dataset.column_names, (
            "Dataset must contain 'transcription' column."
        )

        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.samplerate = samplerate

    def _normalize(self, batch: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        对文本和语言进行标准化
        """
        # 非“任意语言的字母/数字” -> 空格
        NON_TEXT_RE = regex.compile(r"[^\p{L}\p{N}]+")
        WHITESPACE_RE = regex.compile(r"\s+")
        texts = []
        language_codes = []

        transcriptions = batch["transcription"]
        languages = batch["language"]
        for text, lang in zip(transcriptions, languages):
            # 全角半角统一
            text = unicodedata.normalize("NFKC", text)
            # 小写
            text = text.lower()
            # 标点/特殊字符 -> 空格
            text = NON_TEXT_RE.sub(" ", text)
            # 合并空白
            text = WHITESPACE_RE.sub(" ", text)
            # 去首尾空白
            text = text.strip()

            lang = lang.lower()
            lang_code = TO_LANGUAGE_CODE.get(lang, lang)
            if lang_code not in TO_LANGUAGE_CODE.values():
                raise ValueError(f"Unsupported language: {lang}.")

            texts.append(text)
            language_codes.append(lang_code)

        batch["transcription"] = texts
        batch["language"] = language_codes
        return batch

    def _tokenize(self, example: dict[str, Any]) -> dict[str, Any]:
        # 由于需要根据 language 构造，不 batch
        # whisper feature_extractor 在处理时就将音频自动 pad 到了 30s
        example["input_features"] = self.feature_extractor(
            example["audio"]["array"], sampling_rate=self.samplerate
        ).input_features[0]
        transcription_ids = self.tokenizer.encode(
            example["transcription"], add_special_tokens=False
        )
        example["labels"] = (
            self._get_prefix_tokens(example["language"])
            + transcription_ids
            + [self.tokenizer.eos_token_id]
        )
        return example

    def _get_prefix_tokens(self, language: str) -> List[int]:
        prefix_tokens = [
            self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
            self.tokenizer.convert_tokens_to_ids(f"<|{language}|>"),
        ]
        if self.tokenizer.task is not None:
            prefix_tokens.append(
                self.tokenizer.convert_tokens_to_ids(f"<|{self.tokenizer.task}|>")
            )
        if not self.tokenizer.predict_timestamps:
            prefix_tokens.append(
                self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            )
        return prefix_tokens  # type: ignore

    def get_dataset(self, num_proc=0) -> Dataset:
        """
        返回的数据集包含以下列：["input_featrues", "labels"]
        """
        self.dataset = self.dataset.map(
            self._normalize,
            batched=True,
            num_proc=num_proc,
        )
        self.dataset = self.dataset.cast_column(
            "audio", Audio(sampling_rate=self.samplerate)
        )
        # 由于需要根据 language 构造，不 batch
        self.dataset = self.dataset.map(
            self._tokenize, remove_columns=self.dataset.column_names
        )
        return self.dataset

    def get_collator(self, dtype: torch.dtype = torch.float32):
        return SimpleCollator(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            dtype=dtype,
        )
