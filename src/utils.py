from typing import Optional
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
import regex
import unicodedata
from torch import nn
from typing import Dict, Any

TASK_TYPES = ["transcribe", "translate"]


def get_prefix_prompt(language: Optional[str], task: Optional[str], predict_timestamps: bool = False) -> str:
    prompt = "<|startoftranscript|>"
    if language is not None:
        language = language.lower()
        language = TO_LANGUAGE_CODE.get(language, language)
        if language not in TO_LANGUAGE_CODE.values():
            raise ValueError(f"Unsupported language: {language}.")
        prompt += f"<|{language}|>"
    if task is not None:
        if task not in TASK_TYPES:
            raise ValueError(f"Unsupported task: {task}. Supported tasks are: {TASK_TYPES}.")
        prompt += f"<|{task}|>"
    if not predict_timestamps:
        prompt += "<|notimestamps|>"
    return prompt


def normalize_text(text: str) -> str:
    # 全角半角统一
    text = unicodedata.normalize("NFKC", text)
    # 小写
    text = text.lower()
    text = regex.sub(r"[<\[][^>\]]*[>\]]", "", text)  # remove words between brackets
    text = regex.sub(r"\(([^)]+?)\)", "", text)  # remove words between parenthesis
    # text = regex.sub(r"[^\p{L}\p{N}]+", " ", text)  # 标点/特殊字符 -> 空格
    text = regex.sub(r"\s+", " ", text)  # 合并空白
    text = text.strip()  # 去首尾空白
    return text


def print_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
    trainable_params = 0
    total_params = 0

    # model.parameters() 默认会去重共享/权重绑定的 Parameter
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_ratio = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    stats = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_ratio,
    }

    print(
        f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable: {trainable_ratio:.4f}%"
    )
    return stats
