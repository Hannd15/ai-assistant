
# Imports
import random
import subprocess
import os

import platform
import pandas as pd
import unidecode
import threading
from collections import deque
import mss
import cv2

import pyaudio
import numpy as np
from openwakeword.model import Model

from TTS.api import TTS
import sounddevice as sd
import io
import soundfile as sf

import requests
import wave
import audioread
import time

from translate import Translator

from datetime import datetime

from huggingface_hub import InferenceClient
import json


client = InferenceClient(api_key="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")

messages = [
	{
		"role": "user",
		"content":
		"""Desde ahora en adelante y hasta que se te indique lo contrario debes actuar con la
		personalidad de Mr. House. uno de los personajes de Fallout: New Vegas y mantenerte
		dentro del rol, es decir, no hables como si actuaras como MR. House, habla como si
		fueras Mr. House. Más exáctamente eres una IA asistente que debe actuar como si fueras
		el Mr House original cuando se te hable pero manteniendo tu proposito de asistir a la persona que te hable.
		Todas tus respuestas son en el lenguaje español. No uses carácteres especiales en tu respuesta.Deberás responder de forma breve, sin realizar largas listas de viñetas o exceder los los 100 tokens. Todas tus respuestas deben
		ser en texto plano, esto significa sin saltos de linea o estilizado de ningún tipo.
		Todas tus respuestas deben ser conclusivas, es decir, no le preguntes al usuario si
		quiere o necesita más información. Deberás responder de forma breve, sin realizar largas
		listas de viñetas o exceder los los 100 tokens. Aunque estás tomando el rol de Mr. House, si el usuario hace una pregunta informativa debes darle una respuesta lo más acercada a la realidad con un poco de flavor text apropiado.
		Tus respuestas deben ser cortas, concisas y directo al punto a menos que el usuario
		pida información a detalle. Intenta en lo más posible responder en una solo oración
		a menos que el usuario especifique que quiere información detallada."""

	}
]
response = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
	messages=messages,
	max_tokens=500,
	stream=False
)
initial_prompt_confirmation = dict(response)["choices"][0]["message"]
messages.append(initial_prompt_confirmation)

def write_mistral(prompt: str):
    message = {
        "role":"user",
        "content":prompt
    }
    messages.append(message)

    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    	messages=messages,
    	max_tokens=100,
    	stream=False
    )
    response_message = dict(response)["choices"][0]["message"]
    messages.append(response_message)
    return response_message["content"]


# Micrófono
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

# APIs
TTS_URL = "http://191.239.119.23:5000/tts"
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}
WAKEWORD_THRESHOLD = 0.06

GREET_VOICELINES_PATH = "./hous_pregenerated_voicelines/greet/"
CLIP_VOICELINES_PATH = "./hous_pregenerated_voicelines/clip/"
GOODBYE_VOICELINES_PATH = "./hous_pregenerated_voicelines/goodbye/"
WHOAMI_VOICELINES_PATH = "./hous_pregenerated_voicelines/whoami/"
MUSIC_SEARCH_GREET_VOICELINES_PATH = "./hous_pregenerated_voicelines/music_greet/"
MUSIC_SEARCH_VOICELINES_PATH = "./hous_pregenerated_voicelines/music_search/"
SEARCH_VOICELINES_PATH = "./hous_pregenerated_voicelines/search_start/"

# TTS
#tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
#speaker_wav = "FNV_MrHouse.wav"  # Path to a sample of the target speaker's voice

# WakeWord
owwModel = Model(wakeword_models=['house.tflite'], inference_framework='tflite')

# Configurable parameters
# Automatically detect screen resolution
with mss.mss() as sct:
    screen_info = sct.monitors[1]  # Primary monitor
    WIDTH, HEIGHT = screen_info["width"], screen_info["height"]
FPS = 30                          # Frames per second
BUFFER_SECONDS = 30               # Duration of recording buffer in seconds

# Initialize the rolling buffer
frame_buffer = deque(maxlen=BUFFER_SECONDS * FPS)
buffer_lock = threading.Lock()    # Lock to safely copy the buffer


months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
def get_city_time(timezone):
    try:
        # Format city name for URL (e.g., New York as America/New_York)
        timezone = timezone.replace(" ", "_")
        url = f"https://timeapi.io/api/time/current/zone?timezone={timezone}"

        # Send request to the World Time API
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse response data
        data = response.json()
        answer = f"Son las {data['hour']} horas con {data['minute']} minutos del {data['day']} de {months[data['month'] - 1]} del {data['year']}"
        return answer


    except (requests.RequestException, KeyError) as e:
        # Handle errors by falling back to the device's local time
        print(f"Could not get time for {timezone.replace('_', ' ')}. Error: {e}")
        month = datetime.now().strftime("%m")
        device_time = datetime.now().strftime(f"Son las %H horas con %M minutos del %d de {months[int(month)-1]} del %Y")
        return device_time

def get_city_date(timezone):
    try:
        # Format city name for URL (e.g., New York as America/New_York)
        timezone = timezone.replace(" ", "_")
        url = f"https://timeapi.io/api/time/current/zone?timezone={timezone}"

        # Send request to the World Time API
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse response data
        data = response.json()
        answer = f"Hoy es el {data['day']} de {months[data['month'] - 1]} del {data['year']}"
        return answer


    except (requests.RequestException, KeyError) as e:
        # Handle errors by falling back to the device's local time
        print(f"Could not get time for {timezone.replace('_', ' ')}. Error: {e}")
        month = datetime.now().strftime("%m")
        device_time = datetime.now().strftime(f"Hoy es el %d de {months[int(month)-1]} del %Y")
        return device_time

# Definimos la URL de la API
ip  = requests.get('https://api.ipify.org').text
url = f"http://ip-api.com/json/{ip}?lang=es"

try:
    # Realizamos la solicitud
    response = requests.get(url)
    data = response.json()
    USER_CITY = data['city']
    USER_TIMEZONE = data['timezone']
except requests.RequestException as e:
    print("Error al obtener la información de ubicación:", e)
    USER_CITY = None



def get_weather(city):
    if city is None:
        print("No se pudo obtener la ubicación del usuario.")
        return
    url = f"http://api.weatherapi.com/v1/current.json?key=162f1488865e46b59d8205729242810&q={city}&aqi=no"
    response = requests.get(url).json()

    text = f"El clima en {response['location']['name']} es {translator.translate(response['current']['condition']['text'])} con una temperatura de {response['current']['temp_c']} grados centígrados."
    return (text)

import yt_dlp
import os
import pygame

def download_music(query):

    """Busca y descarga el primer resultado de YouTube en MP3."""
    # Crea la carpeta music_download si no existe
    music_folder = 'music_download'
    if not os.path.exists(music_folder):
        os.makedirs(music_folder)

    # Define la plantilla de salida para que use el nombre de la consulta
    ydl_opts = {
        #'ffmpeg_location': r'C:\Users\danie\Desktop\ffmpeg\bin',  # Cambia esta ruta si es necesario
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(music_folder, f'{(query).replace(" ","_")}'),  # Guarda con el nombre de la consulta
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"ytsearch:{query}"])

def play_downloaded_music(query):
    """Reproduce el archivo MP3 descargado usando pygame."""
    # Crea la ruta del archivo usando el nombre de la consulta
    file_path = 'music_download/' +f'{(query).replace(" ","_")}.mp3'

    if os.path.exists(file_path):
        if platform.system() == 'Darwin': # macOS
            subprocess.call(('open', file_path))
        elif platform.system() == 'Windows': # Windows
            subprocess.call(('start', file_path), shell=True)
        else: # linux variants
            subprocess.call(('xdg-open', file_path))
    else:
        print(f"No se encontró el archivo: {file_path}")

import shutil

def delete_music_folder():
    """Elimina la carpeta music_download y su contenido."""
    music_folder = 'music_download'
    if os.path.exists(music_folder):
        shutil.rmtree(music_folder)
        print("Carpeta music_download eliminada.")
    else:
        print("La carpeta music_download no existe.")

def capture_screen():
    """Continuously capture screen and add frames to the buffer."""
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}
        while True:
            start_time = time.time()

            # Capture screen
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Add frame to buffer with lock
            with buffer_lock:
                frame_buffer.append(frame)

            # Wait to achieve desired FPS
            time.sleep(max(1.0 / FPS - (time.time() - start_time), 0))

def save_video(filename="clip"):
    """Save frames from the buffer to a video file."""
    with buffer_lock:
        frames_to_save = list(frame_buffer)

    if not os.path.exists("./clips"):
        os.makedirs("clips")
    filename = "./clips/"+filename+"-"+time.strftime("%Y%m%d-%H%M%S")+".mp4"
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, FPS, (WIDTH, HEIGHT))

    # Write buffered frames to video
    for frame in frames_to_save:
        out.write(frame)

    out.release()
    print(f"Saved recording to {filename}")

# Run the screen capture in a separate thread
capture_thread = threading.Thread(target=capture_screen, daemon=True)
capture_thread.start()

# Keywords
keyword_actions = ["clip", "search","weather", "time","date", "whoami", "music"]
keywords = [
    ["clip", "busc", "clima", "hora", "fecha", "eres", "cancion"],
    [None, "busq", None, None, "dia", None, "musica"],
    [None, "consulta", None, None, "mes", None, "reproducir"],
    [None, None, None, None, "ano", None, "sonido"],
]
keyword_df = pd.DataFrame(keywords, columns=keyword_actions)

def search_keyword(text):
    text = unidecode.unidecode(text)
    text = text.lower()
    for index, row in keyword_df.iterrows():
        for i in keyword_actions:
            if row[i] is not None and row[i] in text:
                return i
    return None

def record_audio_chunk(filename, duration):
    """Graba un fragmento de audio y lo guarda en un archivo WAV."""
    print("Grabando audio...")
    # Graba el audio
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()  # Espera a que la grabación termine

    # Guarda el audio en un archivo WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes para int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    return filename

def transcribe_audio(wav_buffer):
    with open(wav_buffer, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def whisper_listen_and_intent(duration):
    # Grabar y transcribir cada chunk de audio
    audio_chunk = record_audio_chunk(filename="buffer.wav", duration=duration)
    print("Procesando...")
    text = transcribe_audio(audio_chunk)['text']

    text = unidecode.unidecode(text)
    text = text.lower()

    intent = search_keyword(text)
    if intent is not None:
        act_on_intent(intent)

def whisper_listen_once(duration):
    # Grabar y transcribir cada chunk de audio
    audio_chunk = record_audio_chunk(filename="buffer.wav", duration=duration)
    text = transcribe_audio(audio_chunk)['text']
    text = unidecode.unidecode(text)
    text = text.lower()
    return text

def act_on_intent(intent):
    print("Intent: "+intent)
    if intent == "clip":
        save_video()
        play_audio_file(select_random_file_from_folder(CLIP_VOICELINES_PATH))
    elif intent == "search":
        play_audio_file(select_random_file_from_folder(SEARCH_VOICELINES_PATH),blocking=False)
        message = write_mistral(whisper_listen_once(6))
        tts(message)
    elif intent == "weather":
        tts(get_weather(USER_CITY))
    elif intent == "time":
        tts(get_city_time(USER_TIMEZONE))
    elif intent == "date":
        tts(get_city_date(USER_TIMEZONE))
    elif intent == "whoami":
        play_audio_file(select_random_file_from_folder(WHOAMI_VOICELINES_PATH))
    elif intent == "music":
        play_audio_file(select_random_file_from_folder(MUSIC_SEARCH_GREET_VOICELINES_PATH))
        messageMusic = whisper_listen_once(5)  # Obtiene el título o frase para la búsqueda
        play_audio_file(select_random_file_from_folder(MUSIC_SEARCH_VOICELINES_PATH),blocking=False)
        download_music(messageMusic)
        play_downloaded_music(messageMusic)

def tts(text):
    body = {"text":text}
    print("Procesando respuesta")
    response = requests.post(TTS_URL,json=body)

    if response.status_code == 200:
        # Load audio data from response content
        audio_data, samplerate = sf.read(io.BytesIO(response.content), dtype='int16')

        # Play the audio
        sd.play(audio_data, samplerate=samplerate)
        sd.wait()
    else:
        print("Failed to get TTS audio:", response.status_code, response.text)


# Flag to check if the intent function is currently running
is_processing_intent = threading.Event()

def process_intent():
    """Function to handle intent processing in a separate thread."""
    play_audio_file(select_random_file_from_folder(GREET_VOICELINES_PATH))
    whisper_listen_and_intent(3)
    print("Intent processing completed.")

def wakeword_loop():
    print("listening...")
    predicted = False
    intent_thread = None  # Initialize intent_thread to None

    while True:
        # Capture audio from microphone
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Predict using the wake word model
        prediction = owwModel.predict(audio)['house']

        # Check if the wake word threshold is met and not already processing intent
        if prediction > WAKEWORD_THRESHOLD and not is_processing_intent.is_set():
            predicted = True
            is_processing_intent.set()  # Mark as processing intent
            print("Wake word detected. Processing intent...")

            # Start the intent processing in a new thread
            intent_thread = threading.Thread(target=process_intent)
            intent_thread.start()

        # Check if intent_thread exists and is done
        if intent_thread and not intent_thread.is_alive() and predicted:
            # Reset flags once intent processing is complete
            is_processing_intent.clear()
            predicted = False
            print("listening...")  # Ready to listen for the next wake word

def play_audio_file(filename,  blocking=True):
    with wave.open(filename, 'rb') as wf:
        audio_data = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    sd.play(audio_array, samplerate=wf.getframerate(), blocking=blocking)

def select_random_file_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return folder_path + (random.choice(files))

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    delete_music_folder()
    wakeword_loop()
    delete_music_folder()
