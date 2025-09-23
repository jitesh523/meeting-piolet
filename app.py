from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from rag import RAGPipeline
from agent import MeetingAgent
from stt import AudioTranscriber

app = Flask(__name__)
socketio = SocketIO(app)
rag = RAGPipeline()
agent = MeetingAgent(rag)
transcriber = AudioTranscriber()

@app.route('/')
def index():
    return render_template('popup.html')

@socketio.on('question')
def handle_question(data):
    question = data['question']
    answer = agent.process_question(question)
    socketio.emit('answer', {'answer': answer})

@socketio.on('transcript')
def handle_transcript(data):
    rag.index_transcript(data['text'])
    socketio.emit('transcript_update', {'text': data['text']})

if __name__ == "__main__":
    transcriber.start()
    socketio.run(app, debug=True)