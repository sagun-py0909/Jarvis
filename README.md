# Jarvis Voice Assistant

Jarvis is a local, privacy-focused voice assistant powered by open-source speech-to-text (Whisper), text-to-speech (Coqui TTS), and large language models (Ollama/Mistral). It features wake word detection, conversational memory via a vector database (RAG), and natural voice interaction.

## Features
- **Wake Word Detection:** Listens for "Jarvis" using Porcupine.
- **Speech-to-Text:** Transcribes your voice using Whisper (GPU-accelerated).
- **Conversational LLM:** Uses Ollama (Mistral) for intelligent replies.
- **Text-to-Speech:** Speaks responses using Coqui TTS.
- **Memory (RAG):** Stores and retrieves past conversations for context-aware answers.
- **Local & Private:** All processing is done locally; no cloud required.

## How It Works
1. **Say "Jarvis"** to activate.
2. **Speak your query.**
3. Jarvis transcribes, retrieves relevant context, sends to LLM, and speaks the reply.
4. All interactions are stored in a local vector database for improved future responses.

## Requirements
- Python 3.11+
- NVIDIA GPU (for best performance)
- [Ollama](https://ollama.com/) running locally
- Required Python packages: `pvporcupine`, `pvrecorder`, `faster-whisper`, `torch`, `TTS`, `sounddevice`, `faiss-cpu`, `sentence-transformers`

## Setup
1. Clone this repo.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start Ollama and download a model (e.g., `mistral:7b`).
4. Run `main.py`:
   ```
   python main.py
   ```

## File Structure
- `main.py` — Main app logic
- `stt.py` — Speech-to-text utilities
- `tts.py` — Text-to-speech utilities
- `llm.py` — LLM interaction and RAG integration
- `vector_db.py` — Vector database for conversational memory
- `.gitignore` — Files to exclude from git

## Credits
- [Porcupine](https://picovoice.ai/) for wake word
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) for STT
- [Coqui TTS](https://github.com/coqui-ai/TTS) for TTS
- [Ollama](https://ollama.com/) for LLM
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings

---
**Local, private, and open-source voice AI.**
