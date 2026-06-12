import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import librosa
import regex
import torch
from datasets import Audio, Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE


NON_TEXT_RE = regex.compile(r"[^\p{L}\p{N}]+")
WHITESPACE_RE = regex.compile(r"\s+")


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
        input_features = torch.tensor([feature["input_features"] for feature in features], dtype=self.dtype)

        batch = self.tokenizer.pad(
            {"input_ids": [feature["labels"] for feature in features]},
            return_tensors="pt",
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
        transcription_ids = self.tokenizer.encode(example["transcription"], add_special_tokens=False)
        example["labels"] = (
            self._get_prefix_tokens(example["language"]) + transcription_ids + [self.tokenizer.eos_token_id]
        )
        return example

    def _get_prefix_tokens(self, language: str) -> List[int]:
        prefix_tokens = [
            self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
            self.tokenizer.convert_tokens_to_ids(f"<|{language}|>"),
        ]
        if self.tokenizer.task is not None:
            prefix_tokens.append(self.tokenizer.convert_tokens_to_ids(f"<|{self.tokenizer.task}|>"))
        if not self.tokenizer.predict_timestamps:
            prefix_tokens.append(self.tokenizer.convert_tokens_to_ids("<|notimestamps|>"))
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
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=self.samplerate))
        # 由于需要根据 language 构造，不 batch
        self.dataset = self.dataset.map(self._tokenize, remove_columns=self.dataset.column_names)
        return self.dataset

    def get_collator(self, dtype: torch.dtype = torch.float32):
        return SimpleCollator(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            dtype=dtype,
        )


@dataclass
class Collator:
    """
    对动态切出的原始音频做 feature extraction，并 pad labels。
    """

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer
    dtype: torch.dtype
    samplerate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        inputs = self.feature_extractor(
            [feature["audio"] for feature in features],
            sampling_rate=self.samplerate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(dtype=self.dtype)

        batch = self.tokenizer.pad(
            {"input_ids": [feature["labels"] for feature in features]},
            return_tensors="pt",
        )
        labels = batch.input_ids.masked_fill(batch.attention_mask.ne(1), -100)

        return {"input_features": input_features, "labels": labels}


class DataManager(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        audio_path: str | Path,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        samplerate: int = 16000,
        packing_p: float = 0.25,
        packing_max_s: float = 30.0,
    ) -> None:
        """
        dataset 必须包含这些列：['language', 'transcription', 'start_time', 'end_time']
        """
        required_columns = {"language", "transcription", "start_time", "end_time"}
        missing_columns = required_columns.difference(dataset.column_names)
        if missing_columns:
            raise ValueError(f"Dataset missing columns: {sorted(missing_columns)}.")
        if not 0 <= packing_p <= 1:
            raise ValueError("packing_p must be between 0 and 1.")
        if packing_max_s <= 0:
            raise ValueError("packing_max_s must be positive.")

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.samplerate = samplerate
        self.packing_p = packing_p
        self.packing_max_s = packing_max_s
        self.audio_path = Path(audio_path)
        self.audio, _ = librosa.load(self.audio_path, sr=samplerate, mono=True)
        self.records = self._build_records(dataset)
        if not self.records:
            raise ValueError("Dataset is empty.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += len(self.records)
        if idx < 0 or idx >= len(self.records):
            raise IndexError(idx)

        end_idx = self._get_packed_end_idx(idx)
        packed_records = self.records[idx : end_idx + 1]
        start_time = packed_records[0]["start_time"]
        end_time = packed_records[-1]["end_time"]
        transcription = " ".join(record["transcription"] for record in packed_records if record["transcription"])
        language = packed_records[0]["language"]

        labels = self._get_prefix_tokens(language)
        labels += self.tokenizer.encode(transcription, add_special_tokens=False)
        labels += [self.tokenizer.eos_token_id]

        return {
            "audio": self._slice_audio(start_time=start_time, end_time=end_time),
            "labels": labels,
        }

    def _build_records(self, dataset: Dataset) -> List[Dict[str, Any]]:
        records = []
        for idx, example in enumerate(dataset):
            start_time = float(example["start_time"])  # type: ignore
            end_time = float(example["end_time"])  # type: ignore
            if end_time <= start_time:
                raise ValueError(f"Invalid timestamp at index {idx}: {start_time} -> {end_time}.")
            if end_time - start_time > self.packing_max_s:
                raise ValueError(
                    f"Segment at index {idx} is longer than packing_max_s "
                    f"({end_time - start_time:.2f}s > {self.packing_max_s:.2f}s)."
                )

            transcription, language = self._normalize_text_and_language(
                text=example["transcription"],  # type: ignore
                language=example["language"],  # type: ignore
            )
            records.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "language": language,
                    "transcription": transcription,
                }
            )

        records.sort(key=lambda record: record["start_time"])
        return records

    def _normalize_text_and_language(self, text: str, language: str) -> tuple[str, str]:
        # 全角半角统一
        text = unicodedata.normalize("NFKC", str(text))
        # 小写
        text = text.lower()
        # 标点/特殊字符 -> 空格
        text = NON_TEXT_RE.sub(" ", text)
        # 合并空白
        text = WHITESPACE_RE.sub(" ", text)
        # 去首尾空白
        text = text.strip()

        language = str(language).lower()
        language_code = TO_LANGUAGE_CODE.get(language, language)
        if language_code not in TO_LANGUAGE_CODE.values():
            raise ValueError(f"Unsupported language: {language}.")

        return text, language_code

    def _get_packed_end_idx(self, idx: int) -> int:
        start_time = self.records[idx]["start_time"]
        end_idx = idx
        while end_idx + 1 < len(self.records):
            next_record = self.records[end_idx + 1]
            if next_record["end_time"] - start_time > self.packing_max_s:
                break
            if random.random() >= self.packing_p:
                break
            end_idx += 1
        return end_idx

    def _slice_audio(self, start_time: float, end_time: float):
        start = max(0, round(start_time * self.samplerate))
        end = min(len(self.audio), round(end_time * self.samplerate))
        if end <= start:
            raise ValueError(f"Invalid audio slice: {start_time} -> {end_time}.")
        return self.audio[start:end]

    def _get_prefix_tokens(self, language: str) -> List[int]:
        prefix_tokens = [
            self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
            self.tokenizer.convert_tokens_to_ids(f"<|{language}|>"),
        ]
        if self.tokenizer.task is not None:
            prefix_tokens.append(self.tokenizer.convert_tokens_to_ids(f"<|{self.tokenizer.task}|>"))
        if not self.tokenizer.predict_timestamps:
            prefix_tokens.append(self.tokenizer.convert_tokens_to_ids("<|notimestamps|>"))
        return prefix_tokens  # type: ignore

    def get_dataset(self, num_proc: int = 0):
        return self

    def get_collator(self, dtype: torch.dtype = torch.float32):
        return Collator(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            dtype=dtype,
            samplerate=self.samplerate,
        )
