# 准备

```bash
uv sync

. .venv/bin/activate

hf download openai/whisper-large-v3-turbo
```

## 数据集

1. jsonl 数据

类似以下格式，其中`audio`字段是音频路径（相对于该 jsonl 文件）

```jsonl
{"audio": "audio/000001.wav", "language": "zh", "transcription": "塔菲去哪儿了"}
{"audio": "audio/000002.wav", "language": "zh", "transcription": "砰，塔菲出现了"}
```

2. huggingface dataset

至少包含`audio`，`language`，`transcription`这三列，例如：https://huggingface.co/datasets/KYOU-0/Ace-Taffy-voice

# Finetune

1. 全参数微调 decoder + lm_head（encoder 以外部分）

```bash
# 本地数据集
python -m scripts.simple_ft --model_name="openai/whisper-large-v3-turbo" --data_file="./outputs/data/metadata.jsonl"
# huggingface 数据集
python -m scripts.simple_ft --model_name="openai/whisper-large-v3-turbo" --data_file="KYOU-0/Ace-Taffy-voice"
```

2. 微调 lora

```bash
# 微调 Lora
# 本地数据集
python -m scripts.simple_lora --model_name="openai/whisper-large-v3-turbo" --data_file="./outputs/data/metadata.jsonl"
# huggingface 数据集
python -m scripts.simple_lora --model_name="openai/whisper-large-v3-turbo" --data_file="KYOU-0/Ace-Taffy-voice"
```

# Result

数据集：KYOU-0/Ace-Taffy-voice

| model                  |   loss | cer\* |
| ---------------------- | -----: | ----: |
| whisper-large-v3-turbo | 1.5514 | 63.45 |
| simple_ft              | 0.7039 | 44.15 |
| simple_lora            | 0.6937 | 33.14 |

\*去除空白、标点、特殊符号
