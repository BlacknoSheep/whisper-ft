"""
生成 srt 字幕文件
"""

import argparse
from pathlib import Path
import librosa
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import opencc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="generate_srt")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--audio_path", type=str, default="./downloads/audio.wav")
    parser.add_argument("--output_path", type=str, default="./outputs/data/audio.srt")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--batch_size", type=int, default=128)

    return parser.parse_args()


samplerate = 16000

# vad
threshold = 0.2
min_speech_duration_ms = 500

# stt
language = "zh"
device = torch.device("cuda")
dtype = torch.bfloat16


def sample2timestamp(sample: int, sr: int = 16000) -> str:
    milliseconds = round(sample / sr * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def main():
    args = parse_args()

    vad = load_silero_vad(onnx=True)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=device,
    )
    model.eval()

    audio, sr = librosa.load(args.audio_path, sr=samplerate)

    timestamps = get_speech_timestamps(
        torch.from_numpy(audio),
        vad,
        threshold=threshold,
        sampling_rate=samplerate,
        min_speech_duration_ms=min_speech_duration_ms,
    )  # samples
    print(len(timestamps))

    # 生成 srt 字幕文件
    # batch stt
    num_segments = len(timestamps)
    num_batches = (num_segments + args.batch_size - 1) // args.batch_size
    counter = 1
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        for batch_idx in tqdm(range(num_batches)):
            batch_timestamps = timestamps[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            batch_segments = []
            for t in batch_timestamps:
                start, end = t["start"], t["end"]
                segment = audio[start:end]
                batch_segments.append(segment)

            inputs = processor(batch_segments, sampling_rate=16000, return_tensors="pt")
            inputs = {
                k: v.to(device=device, dtype=model.dtype) if v.is_floating_point() else v.to(device)
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                # max_length=448，但是由于 special tokens，实际可用长度要需要短一点
                generated_ids = model.generate(**inputs, language=language, task="transcribe", max_new_tokens=224)
            texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for t, text in zip(batch_timestamps, texts):
                start, end = t["start"], t["end"]
                text = text.strip()
                text = opencc.OpenCC("t2s").convert(text)
                f.write(f"{counter}\n")
                f.write(f"{sample2timestamp(start)} --> {sample2timestamp(end)}\n")
                f.write(f"{text}\n\n")
                counter += 1


if __name__ == "__main__":
    main()
