from flask import Flask, request, jsonify, send_file
from TTS.api import TTS
from flask.helpers import make_response
import numpy as np
import sounddevice as sd
import datetime
import requests
from http import HTTPStatus
import os
import yt_dlp
import io
import soundfile as sf


app = Flask(__name__)

tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
speaker_wav = "FNV_MrHouse.wav"

@app.post('/tts')
def tts():
    if not request.json or "text" not in request.json:
        return make_response("", HTTPStatus.BAD_REQUEST)

    text = request.json["text"]
    audio_array = tts_model.tts(text, speaker_wav=speaker_wav, language='es')

    # Convert audio to a WAV format and save in an in-memory buffer
    wav_io = io.BytesIO()
    sf.write(wav_io, audio_array, samplerate=22050, format="WAV")
    wav_io.seek(0)  # Move to the start of the BytesIO buffer

    return send_file(wav_io, mimetype="audio/wav")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
