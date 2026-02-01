"""
Qwen3-ASR Streaming Inference Demo with vLLM Backend
Works on macOS (with vllm-metal) and Linux/Windows (with CUDA)
"""
import os
import sys

# Disable torch.compile on macOS to avoid inductor errors
if sys.platform == "darwin":
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample audio to 16kHz using linear interpolation."""
    if sr == 16000:
        return wav.astype(np.float32, copy=False)
    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


if __name__ == '__main__':
    is_macos = sys.platform == "darwin"
    has_cuda = torch.cuda.is_available()

    print(f"Platform: {sys.platform}")
    print(f"CUDA available: {has_cuda}")

    # Build model kwargs - omit gpu_memory_utilization on macOS/CPU
    model_kwargs = {
        "model": "Qwen/Qwen3-ASR-0.6B",
        "max_new_tokens": 32,  # Small value for streaming
    }

    if has_cuda:
        model_kwargs["gpu_memory_utilization"] = 0.7

    print("Loading Qwen3-ASR model with vLLM backend...")
    model = Qwen3ASRModel.LLM(**model_kwargs)

    print("Model loaded. Starting streaming transcription...")
    print("-" * 50)

    # Load and prepare audio
    audio_file = "test_audio_en.wav"
    print(f"Transcribing: {audio_file}")
    wav, sr = sf.read(audio_file, dtype="float32")
    wav16k = resample_to_16k(wav, sr)

    # Initialize streaming state
    state = model.init_streaming_state(
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=2.0,
    )

    # Stream audio in 500ms chunks
    step_ms = 500
    step_samples = int(round(step_ms / 1000.0 * 16000))
    pos = 0
    call_id = 0

    print("Streaming output:")
    while pos < wav16k.shape[0]:
        chunk = wav16k[pos:pos + step_samples]
        pos += len(chunk)
        call_id += 1

        model.streaming_transcribe(chunk, state)
        print(f"  [{call_id:03d}] lang={state.language!r} text={state.text!r}")

    # Finalize
    model.finish_streaming_transcribe(state)
    print("-" * 50)
    print(f"Final result:")
    print(f"  Language: {state.language}")
    print(f"  Text: {state.text}")
    print("Streaming transcription complete!")
