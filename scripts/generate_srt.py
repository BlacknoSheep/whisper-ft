"""
生成 srt 字幕文件
"""

from pathlib import Path
import librosa
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import opencc

AUDIO_PATH = "./downloads/audio.wav"
OUTPUT_PATH = "./downloads/my-dataset/audio.srt"

language = "zh"
batch_size = 128
device = torch.device("cuda")
dtype = torch.bfloat16

vad = load_silero_vad(onnx=True)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3-turbo", local_files_only=True
)
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo",
    local_files_only=True,
    dtype=dtype,
    attn_implementation="flash_attention_4",
    device_map=device,
)
model.eval()

audio, sr = librosa.load(AUDIO_PATH, sr=16000)
print(audio.shape, sr)

timestamps = get_speech_timestamps(
    torch.from_numpy(audio), vad, threshold=0.2
)  # samples
print(len(timestamps))


def sample2timestamp(sample: int, sr: int = 16000) -> str:
    milliseconds = round(sample / sr * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


# 生成 srt 字幕文件
# batch stt
num_segments = len(timestamps)
num_batches = (num_segments + batch_size - 1) // batch_size
counter = 1
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    for batch_idx in tqdm(range(num_batches)):
        batch_timestamps = timestamps[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]
        batch_segments = []
        for t in batch_timestamps:
            start, end = t["start"], t["end"]
            segment = audio[start:end]
            batch_segments.append(segment)

        inputs = processor(batch_segments, sampling_rate=16000, return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=model.dtype)
            if v.is_floating_point()
            else v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            # max_length=448，但是由于 special tokens，实际可用长度要需要短一点
            generated_ids = model.generate(
                **inputs, language=language, task="transcribe", max_new_tokens=224
            )
        texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for t, text in zip(batch_timestamps, texts):
            start, end = t["start"], t["end"]
            text = text.strip()
            text = opencc.OpenCC("t2s").convert(text)
            f.write(f"{counter}\n")
            f.write(f"{sample2timestamp(start)} --> {sample2timestamp(end)}\n")
            f.write(f"{text}\n\n")
            counter += 1
