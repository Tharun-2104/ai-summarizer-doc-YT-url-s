import os
import re
import io
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
# ❌ Remove dotenv import and load_dotenv
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
