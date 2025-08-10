
# stt.py - Speech-to-Text utilities for Jarvis
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import torch
import time

# CONFIG
MODEL_SIZE = "small"           # tiny / base / small / medium / large
SAMPLE_RATE = 16000
DEFAULT_RECORD_SECONDS = 5
MIC_DEVICE_INDEX = 0           # change if your mic is another index

def init_whisper(device_preference="cuda"):
    """
    Load Whisper model (GPU preferred, fallback to CPU).
    Returns: (model, device)
    """
    try:
        if device_preference == "cuda" and torch.cuda.is_available():
            print("🚀 Loading Whisper model on GPU...")
            model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
            print("✅ Whisper model loaded on GPU.")
            return model, "cuda"
        else:
            raise RuntimeError("GPU not available or not preferred")
    except Exception as e:
        print(f"⚠️ GPU load failed / skipped: {e}")
        print("💡 Falling back to CPU (int8)...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("✅ Whisper model loaded on CPU.")
        return model, "cpu"

def record_audio(duration=DEFAULT_RECORD_SECONDS, device=MIC_DEVICE_INDEX, sample_rate=SAMPLE_RATE):
    """
    Record audio from microphone and return (audio, sample_rate).
    """
    print(f"🎤 Recording for {duration} seconds on device {device} (sr={sample_rate})...")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32", device=device)
        sd.wait()
        audio = np.squeeze(audio)
        print(f"🔍 Audio min={float(np.min(audio)):.6f}, max={float(np.max(audio)):.6f}, mean={float(np.mean(audio)):.6e}")
        # save debug file (optional)
        try:
            sf.write("debug.wav", audio, sample_rate)
        except Exception:
            pass
        return audio, sample_rate
    except Exception as e:
        print(f"❌ Audio recording error: {e}")
        return None, sample_rate

def transcribe(audio_data, sample_rate, whisper_model):
    """
    Transcribe audio_data (float32 mono) using the preloaded whisper_model.
    Returns: transcribed text (str)
    """
    if audio_data is None:
        return ""
    try:
        print("📝 Transcription starting...")
        start = time.time()
        # faster-whisper accepts NumPy float32 arrays sampled at sample_rate
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        segments, info = whisper_model.transcribe(audio_data, beam_size=1)
        elapsed = time.time() - start
        print(f"⏱ Inference time: {elapsed:.2f}s, detected language: {info.language}")
        text = " ".join([seg.text.strip() for seg in segments]).strip()
        print(f"✅ Transcription: {text!r}")
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""
