## Meeting Copilot

An AI-powered meeting assistant that transcribes live audio, builds a searchable knowledge base from your meeting transcript and documents, and answers questions in real time via an agentic RAG pipeline.

### Key Features
- Live speech-to-text transcription using Whisper
- Retrieval-Augmented Generation (RAG) over transcripts and uploaded documents (PDF/TXT/DOCX)
- Agentic question answering with tools for contextual retrieval
- Realtime UX backed by Flask + Socket.IO

### Tech Stack
- Python 3.9+
- Flask, Flask-SocketIO
- OpenAI Whisper (CPU by default)
- LangChain (FAISS vector store, HuggingFace embeddings)
- PDF/DOCX parsing with PyPDF2 and python-docx

## Getting Started

### 1) Prerequisites
- Python 3.9 or newer
- FFmpeg installed (required by Whisper)
  - macOS (Homebrew):
    ```bash
    brew install ffmpeg
    ```
  - Ubuntu/Debian:
    ```bash
    sudo apt update && sudo apt install -y ffmpeg
    ```

### 2) Clone the Repository
```bash
git clone https://github.com/jitesh523/meeting-piolet.git
cd meeting-piolet
```

### 3) Create a Virtual Environment and Install Dependencies
It is strongly recommended to keep your virtual environment out of Git (`.gitignore` already excludes `venv/`).

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt  # If you add one, or install manually (see below)
```

If you do not have a `requirements.txt` yet, install the core packages manually:
```bash
pip install flask flask-socketio langchain-community langchain-huggingface faiss-cpu
pip install sentence-transformers PyPDF2 python-docx
pip install openai-whisper pyaudio numpy
```

Note: On macOS, you may need Xcode CLT and PortAudio for `pyaudio`:
```bash
brew install portaudio
pip install pyaudio
```

### 4) Configure Environment

The agent uses a chat model via LangChain. Set your provider API key(s) as environment variables. Example for Google Gemini via LangChain:
```bash
export GOOGLE_API_KEY="<your_gemini_key>"
```

Ensure your code reads from env vars instead of hardcoding secrets.

### 5) Run the App
```bash
python app.py
```
This starts a Flask + Socket.IO server and launches the audio transcriber threads. Navigate to the rendered UI (popup) route at:
```
http://127.0.0.1:5000/
```

## Project Structure
```
.
├── app.py                # Flask + Socket.IO server and event handlers
├── agent.py              # Agent built with LangChain, uses retrieval tool
├── rag.py                # RAGPipeline: indexing, embeddings, FAISS search
├── stt.py                # Whisper-based live audio transcription
├── templates/
│   └── popup.html        # Simple UI to ask questions and view answers
├── README.md
└── .gitignore
```

## How It Works

### 1) Speech-to-Text
`stt.py` spawns two threads:
- Captures microphone audio using PyAudio
- Transcribes buffers with Whisper (`base` model by default)

### 2) RAG Indexing
`rag.py` loads documents (PDF/TXT/DOCX), splits text into chunks with `RecursiveCharacterTextSplitter`, embeds them using `sentence-transformers/all-MiniLM-L6-v2`, and stores vectors in FAISS. Meeting transcript text is incrementally added to the same store.

### 3) Agentic QA
`agent.py` creates a LangChain agent with a retrieval tool that queries the FAISS index. Incoming questions from the client are answered using the model plus retrieved context.

### 4) Realtime UI
`app.py` exposes Socket.IO events:
- `question` → agent answers, emits `answer`
- `transcript` → text is indexed, emits `transcript_update`

## Usage
1) Start the server (`python app.py`).
2) Open the UI and ask a question while the transcript is being built.
3) Optionally upload or point the RAG pipeline at documents to enrich context.

To index documents programmatically:
```python
from rag import RAGPipeline
rag = RAGPipeline()
rag.index_documents(["/path/to/file.pdf", "/path/to/notes.txt"])  # supports .pdf/.txt/.docx
```

## Configuration Tips
- Adjust chunk size/overlap in `RAGPipeline` for your content types.
- Switch Whisper model to `small`, `medium`, or `large-v3` for accuracy, or set device to `cuda` if you have a GPU.
- Replace the chat model/provider in `agent.py` as needed; ensure keys are set via environment variables.

## Security & Privacy
- Do not commit virtual environments, large binaries, or secrets. `.gitignore` is configured to exclude common artifacts.
- Store API keys in environment variables or a secrets manager; never hardcode keys in source.

## Troubleshooting
- Large file push rejected by GitHub: ensure `venv/` is not tracked and purge it from history if necessary.
- `pyaudio` build issues on macOS: install `portaudio` with Homebrew and reinstall `pyaudio`.
- Whisper requires FFmpeg; verify `ffmpeg -version` works.
- FAISS install issues on Apple Silicon: prefer `faiss-cpu` and recent Python.

## Roadmap
- File upload UI for PDFs/DOCX/TXT
- Persisted vector store across sessions
- Speaker diarization and timestamps in transcript
- Summarization and action items at meeting end

## License
This project is provided as-is. Add a license of your choice (e.g., MIT) if you plan to distribute.


## Usage
- Run `python app.py` and open http://localhost:5000
- Upload PDFs to the project directory for RAG
