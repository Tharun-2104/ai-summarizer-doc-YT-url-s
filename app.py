import os
import re
import io
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
# ❌ Removed dotenv import and load_dotenv (Render doesn’t use local .env files)
# from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from youtube_transcript_api import YouTubeTranscriptApi
import docx2txt
import PyPDF2
import pytube
import speech_recognition as sr
from pydub import AudioSegment
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_model():
    api_key = os.getenv("MISTRAL_API_KEY")   # ✅ Reads directly from Render
    if not api_key or api_key.strip() == "" or "YOUR_API_KEY_HERE" in api_key:
        print("DEBUG: MISTRAL_API_KEY missing or invalid")
        return None
    return ChatMistralAI(model="mistral-small-2506", temperature=0.7, api_key=api_key)

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        return "".join(page.extract_text() for page in pdf_reader.pages)

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def transcribe_audio(audio_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception:
        return ""

def extract_video_id(url):
    """
    Extracts the 11-character YouTube video ID from any valid YouTube URL.
    Works for both youtu.be and youtube.com/watch?v= formats.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def get_youtube_transcript(url):
    video_id = extract_video_id(url)
    if not video_id:
        return None, "Invalid YouTube URL format."

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        text = " ".join([t.text for t in transcript])
        return text, None
    except Exception as e:
        try:
            yt = pytube.YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_file_path = os.path.join(tmpdir, "audio.mp4")
                audio_stream.download(output_path=tmpdir, filename="audio.mp4")
                wav_file_path = os.path.join(tmpdir, "audio.wav")
                AudioSegment.from_file(audio_file_path).export(wav_file_path, format="wav")
                transcribed_text = transcribe_audio(wav_file_path)
                if transcribed_text:
                    return transcribed_text, None
                else:
                    return None, f"Transcript unavailable and audio transcription failed: {str(e)}"
        except Exception as audio_err:
            return None, f"Error fetching content: {str(e)} | Audio fallback error: {str(audio_err)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    summary_style = request.form.get('summary_style', 'Executive Summary')
    length_choice = request.form.get('length_choice', 'Medium')
    source_type = request.form.get('source_type')
    final_text = ""

    if source_type == 'file':
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'error': 'No file provided'}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            if filename.endswith('.pdf'):
                final_text = extract_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                final_text = extract_text_from_docx(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    final_text = f.read()
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    elif source_type == 'youtube':
        youtube_url = request.form.get('youtube_url')
        if not youtube_url:
            return jsonify({'error': 'No YouTube URL provided'}), 400
        final_text, error = get_youtube_transcript(youtube_url)
        if error:
            return jsonify({'error': error}), 400

    if not final_text:
        return jsonify({'error': 'No text could be extracted'}), 400

    model = get_model()
    if not model:
        return jsonify({'error': 'Mistral API Key not found.'}), 500

    length_map = {"Very Short": "3-5 sentences", "Medium": "7-10 sentences", "Comprehensive": "15-20 sentences"}
    prompt = f"""
    You are a professional content analyst. Please provide a {summary_style} of the following content.
    The summary should be approximately {length_map.get(length_choice, '7-10 sentences')} in length.

    Content:
    {final_text[:15000]}
    """

    try:
        response = model.invoke(prompt)
        return jsonify({'summary': response.content, 'style': summary_style})
    except Exception as e:
        return jsonify({'error': f'Generation Error: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download():
    content = request.form.get('content')
    if not content:
        return "No content", 400
    buffer = io.BytesIO()
    buffer.write(content.encode('utf-8'))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='summary.txt', mimetype='text/plain')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Flask on port {port}...")
    app.run(host="0.0.0.0", port=port)


