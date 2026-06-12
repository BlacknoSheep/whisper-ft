# Prepare
 
## 环境配置

```bash
uv sync

# 安装 flash-atention-4：https://github.com/Dao-AILab/flash-attention
# 对于 whisper, flash-atention-4 只能用于推理，不能用于训练

# 如果需要上传日志到 wandb
# `--report-to-wandb` 将日志上传到 wandb
wandb login
```

## 下载基础模型

```bash
# 预下载模型，避免每次都需要校验
# hf download <hf_model_name>
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

1. 微调 decoder

如果只是为了提高识别准确率，微调 decoder 足够了，收敛速度非常快。  
特别是 whisper-large-v3-turbo 的 decoder 只有 4 层，显存需求低，效率高。

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
