# main.py
import pvporcupine
from pvrecorder import PvRecorder
import time
from stt import init_whisper, record_audio, transcribe, MODEL_SIZE, MIC_DEVICE_INDEX
from TTS.api import TTS
from llm import chat_with_llm
import sys

# CONFIG - replace these as required
WAKEWORD_PATH = "Jarvis.ppn"                         # your .ppn wakeword file
import os
from dotenv import load_dotenv
load_dotenv()
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "YOUR_KEY_HERE")  # loaded from .env
LLM_API_URL = "http://localhost:11434/api/generate" # Ollama local API
LLM_MODEL = "mistral:7b"                            # model id for Ollama (change if needed)
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"  # Coqui TTS model (can be changed)
MIC_DEVICE_INDEX = MIC_DEVICE_INDEX                 # use same index as stt

# Initialize models (GPU-first)
whisper_model, model_device = init_whisper(device_preference="cuda")

# Init TTS (loads on first use)
print("Loading TTS model (Coqui)...")
tts = TTS(TTS_MODEL)   # can take some time first run
print("✅ TTS loaded.")

    # ...removed local chat_with_llm, now using llm.py version...

def speak(text: str):
    """Synthesize and play TTS (blocking)."""
    if not text:
        return
    print(f"🔊 Jarvis says: {text}")
    try:
        wav = tts.tts(text)
        # TTS returns numpy float array, sample rate usually 22050 or 24000 depending model
        import sounddevice as sd
        sr = tts.synthesizer.output_sample_rate if hasattr(tts, "synthesizer") else 22050
        sd.play(wav, samplerate=sr)
        sd.wait()
    except Exception as e:
        print(f"❌ TTS error: {e}")

def main():
    # Setup porcupine
    porcupine = pvporcupine.create(access_key=PICOVOICE_ACCESS_KEY, keyword_paths=[WAKEWORD_PATH])
    recorder = PvRecorder(device_index=MIC_DEVICE_INDEX, frame_length=porcupine.frame_length)
    print("🎧 Listening for wake word 'Jarvis'... (Ctrl+C to exit)")

    try:
        recorder.start()
        while True:
            pcm = recorder.read()
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("\nWake word detected! Jarvis is listening...")
                audio, sr = record_audio(duration=5, device=MIC_DEVICE_INDEX)
                user_text = transcribe(audio, sr, whisper_model)
                if not user_text:
                    print("⚠️ No transcription; skipping LLM/TTS.")
                    continue

                # Send to LLM
                print("🤖 Sending to LLM...")
                reply = chat_with_llm(user_text)
                print(f"🧠 LLM reply: {reply}")

                # Speak reply
                speak(reply)

                print("\n🎧 Listening for wake word 'Jarvis' again...")
            # small sleep to avoid spinning (Porcupine read is blocking but ok)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Jarvis...")
    finally:
        try:
            recorder.stop()
            porcupine.delete()
            recorder.delete()
        except Exception:
            pass

if __name__ == "__main__":
    main()
