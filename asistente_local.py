
# Imports
import random
import os
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

import requests
import wave
import audioread
import time

from meta_ai_api import MetaAI
from translate import Translator

from datetime import datetime


translator = Translator(from_lang="en", to_lang="es")

ai = MetaAI()

prompt = "Desde ahora en adelante y hasta que se te indique lo contrario debes actuar con la personalidad de Mr. House. uno de los personajes de Fallout: New Vegas. Más exáctamente eres una IA asistente que debe actuar como si fueras el Mr House original cuando se te hable pero manteniendo tu proposito de asistir a la persona que te hable. Todas tus respuestas son en el lenguaje español. Deberás responder de forma breve, no excediendo las 300 palabras para cada consulta que se te haga. Todas tus respuestas deben ser en texto plano, esto significa sin saltos de linea o estilizado de ningún tipo. Todas tus respuestas deben ser conclusivas, es decir, no le preguntes al usuario si quiere o necesita más información "

ai.prompt(message=prompt, new_conversation=True)

# Micrófono
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

# APIs
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}
WAKEWORD_THRESHOLD = 0.06

GREET_VOICELINES_PATH = "./hous_pregenerated_voicelines/greet/"
GOODBYE_VOICELINES_PATH = "./hous_pregenerated_voicelines/goodbye/"
WHOAMI_VOICELINES_PATH = "./hous_pregenerated_voicelines/whoami/"
SEARCH_VOICELINES_PATH = "./hous_pregenerated_voicelines/search_start/"

# TTS
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
speaker_wav = "FNV_MrHouse.wav"  # Path to a sample of the target speaker's voice

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

def write_meta(prompt: str):
    stream = ai.prompt(message=prompt, new_conversation=False)
    return stream['message']

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
    ydl_opts = {
        'ffmpeg_location': r'C:\Users\danie\Desktop\ffmpeg\bin',  # Cambia esta ruta si es necesario
        'format': 'bestaudio/best',
        'outtmpl': 'downloaded_music.mp3',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"ytsearch:{query}"])

def play_downloaded_music():
    """Reproduce el archivo MP3 descargado usando pygame."""
    if os.path.exists('downloaded_music.mp3'):
        pygame.mixer.init()
        pygame.mixer.music.load('downloaded_music.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Espera mientras se reproduce  

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

    filename = filename+"-"+time.strftime("%Y%m%d-%H%M%S")+".mp4"
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
    if intent == "clip":
        save_video()
    elif intent == "search":
        play_audio_file(select_random_file_from_folder(SEARCH_VOICELINES_PATH))
        message = write_meta(whisper_listen_once(5))
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
        messageMusic = whisper_listen_once(5)  # Obtiene el título o frase para la búsqueda
        download_music(messageMusic)
        play_downloaded_music()


def tts(text):
    audio_array = tts_model.tts(text, speaker_wav=speaker_wav, language='es')

    # Play the audio (or save it as needed)
    sd.play(audio_array, samplerate=tts_model.synthesizer.output_sample_rate)
    sd.wait()

import threading

import threading
import time

import threading
import time

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
    wakeword_loop()
    #wakeword_loop()