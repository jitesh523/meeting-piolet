import pyaudio
import whisper
import numpy as np
import queue
import threading

class AudioTranscriber:
    def __init__(self):
        self.model = whisper.load_model("base", device="cpu")  # Use "large-v3" for better accuracy, "cuda" for GPU
        self.audio_queue = queue.Queue()
        self.transcript = []

    def capture_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        print("Capturing audio...")
        while True:
            data = stream.read(1024, exception_on_overflow=False)
            self.audio_queue.put(np.frombuffer(data, dtype=np.int16))

    def transcribe_audio(self):
        while True:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_float = audio_data.astype(np.float32) / 32768.0  # Convert to float
                result = self.model.transcribe(audio=audio_float, language="en", fp16=False)
                if result["text"]:
                    self.transcript.append(result["text"])
                    print(f"Transcript: {result['text']}")

    def start(self):
        threading.Thread(target=self.capture_audio, daemon=True).start()
        threading.Thread(target=self.transcribe_audio, daemon=True).start()

if __name__ == "__main__":
    transcriber = AudioTranscriber()
    transcriber.start()
    input("Press Enter to stop...\n")