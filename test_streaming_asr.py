"""
Qwen3-ASR Streaming Inference Demo with Transformers Backend
Works on macOS/Apple Silicon
"""
import torch
from qwen_asr import Qwen3ASRModel

if __name__ == '__main__':
    print("Loading Qwen3-ASR model with Transformers backend...")

    # Use MPS (Metal) on Mac, or CUDA on Linux/Windows with GPU
    device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",  # Using smaller model for faster loading
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=256,
    )

    print("Model loaded. Starting transcription...")
    print("-" * 50)

    # Transcribe the audio file
    audio_file = "test_audio_en.wav"
    print(f"Transcribing: {audio_file}")

    results = model.transcribe(
        audio=audio_file,
        language=None,  # Auto-detect language
    )

    print(f"Detected Language: {results[0].language}")
    print(f"Transcription: {results[0].text}")
    print("-" * 50)
    print("Transcription complete!")
