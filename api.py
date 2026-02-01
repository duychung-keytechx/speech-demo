"""
Flask API for Qwen3-ASR Streaming Inference
Works on macOS (with vllm-metal) and Linux/Windows (with CUDA)
"""
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

# Disable torch.compile on macOS to avoid inductor errors
if sys.platform == "darwin":
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from qwen_asr import Qwen3ASRModel

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global ASR model
asr: Optional[Qwen3ASRModel] = None

# Streaming parameters
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 2.0


@dataclass
class Session:
    state: object
    created_at: float
    last_seen: float


SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60  # 10 minutes


def _gc_sessions():
    """Garbage collect expired sessions."""
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        try:
            if asr:
                asr.finish_streaming_transcribe(SESSIONS[sid].state)
        except Exception:
            pass
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start a new transcription session."""
    if not asr:
        return jsonify({"error": "Model not loaded"}), 500

    session_id = uuid.uuid4().hex
    state = asr.init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )
    now = time.time()
    SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
    return jsonify({"session_id": session_id})


@app.route("/api/chunk", methods=["POST"])
def api_chunk():
    """Process an audio chunk."""
    if not asr:
        return jsonify({"error": "Model not loaded"}), 500

    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "Invalid session_id"}), 400

    if request.mimetype != "application/octet-stream":
        return jsonify({"error": "Expected application/octet-stream"}), 400

    raw = request.get_data(cache=False)
    if len(raw) % 4 != 0:
        return jsonify({"error": "Float32 bytes length not multiple of 4"}), 400

    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

    asr.streaming_transcribe(wav, s.state)

    return jsonify({
        "language": getattr(s.state, "language", "") or "",
        "text": getattr(s.state, "text", "") or "",
    })


@app.route("/api/finish", methods=["POST"])
def api_finish():
    """Finish a transcription session and get final result."""
    if not asr:
        return jsonify({"error": "Model not loaded"}), 500

    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "Invalid session_id"}), 400

    asr.finish_streaming_transcribe(s.state)
    out = {
        "language": getattr(s.state, "language", "") or "",
        "text": getattr(s.state, "text", "") or "",
    }
    SESSIONS.pop(session_id, None)
    return jsonify(out)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": asr is not None,
    })


def load_model(model_path: str = "Qwen/Qwen3-ASR-1.7B"):
    """Load the ASR model."""
    global asr

    has_cuda = torch.cuda.is_available()
    print(f"Platform: {sys.platform}")
    print(f"CUDA available: {has_cuda}")

    model_kwargs = {
        "model": model_path,
        "max_new_tokens": 32,
    }

    if has_cuda:
        model_kwargs["gpu_memory_utilization"] = 0.7

    print(f"Loading model: {model_path}")
    asr = Qwen3ASRModel.LLM(**model_kwargs)
    print("Model loaded successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-ASR Streaming API")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="Model path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    args = parser.parse_args()

    load_model(args.model)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
