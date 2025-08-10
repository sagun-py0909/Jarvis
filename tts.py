
# Improved TTS module for Jarvis
from TTS.api import TTS
import sounddevice as sd

# Model config
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
_tts_model = None

def get_tts_model():
    global _tts_model
    if _tts_model is None:
        print("Loading TTS model (Coqui)...")
        _tts_model = TTS(TTS_MODEL)
        print("✅ TTS loaded.")
    return _tts_model

def speak(text: str):
    """Synthesize and play TTS (blocking)."""
    if not text:
        return
    print(f"🔊 Jarvis says: {text}")
    try:
        tts_model = get_tts_model()
        wav = tts_model.tts(text)
        sr = tts_model.synthesizer.output_sample_rate if hasattr(tts_model, "synthesizer") else 22050
        sd.play(wav, samplerate=sr)
        sd.wait()
    except Exception as e:
        print(f"❌ TTS error: {e}")
