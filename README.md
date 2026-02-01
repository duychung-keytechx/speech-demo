# Qwen3 ASR Streaming Demo

Real-time speech-to-text using Qwen3-ASR model with vLLM-metal on macOS.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.12
- [vllm-metal](https://github.com/vllm-project/vllm-metal) installed

## Running the Server

### CPU Mode (Recommended for macOS)

```bash
TORCH_COMPILE_DISABLE=1 qwen-asr-demo-streaming \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --host 0.0.0.0 \
  --port 8000
```

The `TORCH_COMPILE_DISABLE=1` flag is required to avoid PyTorch inductor compilation errors on macOS.

### Additional Options

| Option | Description | Default |
|--------|-------------|---------|
| `--asr-model-path` | Path to ASR model | Required |
| `--host` | Host to bind to | `127.0.0.1` |
| `--port` | Port to listen on | `8000` |
| `--gpu-memory-utilization` | GPU memory fraction (not used in CPU mode) | `0.9` |

## API Endpoints

### Web Interface

```
GET /
```

Returns the web UI for real-time transcription.

### Start Session

```
POST /api/start
```

Starts a new transcription session.

**Response:**
```json
{
  "session_id": "a844acb263ee4702b6a716c86274b5b4"
}
```

### Send Audio Chunk

```
POST /api/chunk?session_id=<session_id>
```

Sends an audio chunk for processing.

**Request:**
- Content-Type: `audio/wav` or raw audio bytes
- Body: Audio data chunk

**Response:**
```json
{
  "partial_transcript": "Hello world"
}
```

### Finish Session

```
POST /api/finish?session_id=<session_id>
```

Completes the transcription session and returns final result.

**Response:**
```json
{
  "transcript": "Hello world, this is the final transcription."
}
```

## Example Usage

### Using curl

```bash
# Start a session
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/start | jq -r '.session_id')

# Send audio chunk
curl -X POST "http://localhost:8000/api/chunk?session_id=$SESSION_ID" \
  -H "Content-Type: audio/wav" \
  --data-binary @audio_chunk.wav

# Finish and get final transcript
curl -X POST "http://localhost:8000/api/finish?session_id=$SESSION_ID"
```

### Using Python

```python
import requests

# Start session
response = requests.post("http://localhost:8000/api/start")
session_id = response.json()["session_id"]

# Send audio chunks
with open("audio.wav", "rb") as f:
    chunk = f.read(16000)  # Read 1 second of 16kHz audio
    while chunk:
        requests.post(
            f"http://localhost:8000/api/chunk?session_id={session_id}",
            data=chunk,
            headers={"Content-Type": "audio/wav"}
        )
        chunk = f.read(16000)

# Get final transcript
response = requests.post(f"http://localhost:8000/api/finish?session_id={session_id}")
print(response.json()["transcript"])
```

## Known Issues

### Library Conflict Warning

You may see warnings about duplicate `AVFFrameReceiver` classes between `av` and `cv2` packages. These are harmless and don't affect functionality.

### CPU-Only Mode

Currently runs in CPU mode on macOS. The `--gpu-memory-utilization` flag is ignored as vLLM-metal GPU support for this model architecture is still in development.

## Model Information

- **Model:** Qwen/Qwen3-ASR-1.7B
- **Max sequence length:** 65,536 tokens
- **KV cache:** ~599,168 tokens (with default settings)
