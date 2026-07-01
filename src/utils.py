from typing import Optional
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
import regex
import unicodedata

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
    """
    对文本和语言进行标准化
    """
    # 非“任意语言的字母/数字” -> 空格
    NON_TEXT_RE = regex.compile(r"[^\p{L}\p{N}]+")
    WHITESPACE_RE = regex.compile(r"\s+")

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
    return text
