# Prepare

```bash
uv sync

# 安装 flash-atention-4：https://github.com/Dao-AILab/flash-attention
# 对于 whisper, flash-atention-4 只能用于推理，不能用于训练
```

```bash
# `--report-to-wandb` 将日志上传到 wandb
wandb login
```

```bash
# 预下载模型，避免每次都需要校验
# hf download <hf_model_name>
hf download openai/whisper-large-v3-turbo
```
