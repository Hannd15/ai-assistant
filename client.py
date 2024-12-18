
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
import pygame

from translate import Translator

from datetime import datetime

from huggingface_hub import InferenceClient
import json
import math


client = InferenceClient(api_key="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")

# Prompt inicial
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

# Se envía el prompt y se guarda junto con la respuesta de Mistral
response = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
	messages=messages,
	max_tokens=500,
	stream=False
)
initial_prompt_confirmation = dict(response)["choices"][0]["message"]
messages.append(initial_prompt_confirmation)

def write_mistral(prompt: str):
    """ Se le envía el prompt a Mistral junto con el historial de mensajes """
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


# Configuración del micrófono
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

# APIs
TTS_URL = "http://191.239.119.23:5000/tts"
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
headers = {"Authorization": "Bearer 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"}

# Umbrales de detección de las palabras de activación
WAKEWORD_THRESHOLD = 0.06
SLEEP_THRESHOLD = 0.08

# Rutas de las lineas de voz
GREET_VOICELINES_PATH = "./hous_pregenerated_voicelines/greet/"
CLIP_VOICELINES_PATH = "./hous_pregenerated_voicelines/clip/"
GOODBYE_VOICELINES_PATH = "./hous_pregenerated_voicelines/goodbye/"
WHOAMI_VOICELINES_PATH = "./hous_pregenerated_voicelines/whoami/"
MUSIC_SEARCH_GREET_VOICELINES_PATH = "./hous_pregenerated_voicelines/music_greet/"
MUSIC_SEARCH_VOICELINES_PATH = "./hous_pregenerated_voicelines/music_search/"
SEARCH_VOICELINES_PATH = "./hous_pregenerated_voicelines/search_start/"
WAIT_VOICELINES_PATH = "./hous_pregenerated_voicelines/wait/"
REPEAT_VOICELINES_PATH = "./hous_pregenerated_voicelines/repeat/"

# TTS
#tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
#speaker_wav = "FNV_MrHouse.wav"  # Path to a sample of the target speaker's voice

# Modelo de detección de palabras de activación
owwModel = Model(wakeword_models=['house.tflite','adios.tflite'], inference_framework='tflite')

# Configuración de parametros para la grabación de pantalla
with mss.mss() as sct: # Detección automática de la resolución de la pantalla
    screen_info = sct.monitors[1]
    WIDTH, HEIGHT = screen_info["width"], screen_info["height"]
FPS = 30
BUFFER_SECONDS = 30

# Buffer de grabación
frame_buffer = deque(maxlen=BUFFER_SECONDS * FPS)
buffer_lock = threading.Lock()

import pygame
import sys

import pygame
import sys
import threading

# Initialize pygame
pygame.init()

# Configure the window
ancho, alto = 800, 600
ventana = pygame.display.set_mode((ancho, alto))
pygame.display.set_caption('Mr. House')

# Load images
disconnected = pygame.image.load('images/connection_lost.png')
lit_up = pygame.image.load('images/lit_up_house.png')
neutral = pygame.image.load('images/house.png')
freaky = pygame.image.load('images/freaky_house.jpg')

ventana.fill((255, 255, 255))  # White background

# Global state and lock for thread safety
state = "disconnected"  # Possible states: 'talk', 'disconnected', 'idle'
state_lock = threading.Lock()
running = True

# Functions to change the state
def set_talk():
    global state
    with state_lock:
        state = "talk"

def set_disconnected():
    global state
    with state_lock:
        state = "disconnected"

def set_idle():
    global state
    with state_lock:
        state = "idle"

def stop_pygame():
    global running
    running = False

# Display functions
def show_disconnected():
    ventana.blit(disconnected, (ancho // 2 - disconnected.get_width() // 2, alto // 2 - disconnected.get_height() // 2))

def show_neutral():
    ventana.blit(neutral, (ancho // 2 - neutral.get_width() // 2, alto // 2 - neutral.get_height() // 2))

def show_lit_up():
    ventana.blit(lit_up, (ancho // 2 - lit_up.get_width() // 2, alto // 2 - lit_up.get_height() // 2))

# Main loop
def game_loop():
    global state
    global running
    clock = pygame.time.Clock()
    talk_toggle = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ventana.fill((255, 255, 255))  # Clear the screen

        # Determine the state to display
        with state_lock:
            current_state = state

        if current_state == "talk":
            # Alternate randomly between lit_up and neutral
            show_lit_up()
            pygame.display.flip()
            random_interval = random.randint(50, 300)  # Random interval in milliseconds
            pygame.time.wait(random_interval)

            ventana.fill((255, 255, 255))  # Clear the screen
            show_neutral()
            pygame.display.flip()
            random_interval = random.randint(50, 300)  # Random interval in milliseconds
            pygame.time.wait(random_interval)
        elif current_state == "disconnected":
            show_disconnected()
        elif current_state == "idle":
            show_neutral()

        # Update the display if not in "talk" mode
        if current_state != "talk":
            pygame.display.flip()
            clock.tick(30)  # Limit FPS



months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
def get_city_time(timezone):
    """ Consigue la hora del lugar en que se encuentra el usuario dependiendo de la zona horaria, en caso de un error de conección toma la hora del dispositivo """

    try:
        # Format city name for URL (e.g., New York as America/New_York)
        timezone = timezone.replace(" ", "_")
        url = f"https://timeapi.io/api/time/current/zone?timezone={timezone}"

        # Send request to the World Time API
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse response data
        data = response.json()
        answer = f"Son las {data['hour']} horas con {data['minute']} minutos"
        return answer


    except (requests.RequestException, KeyError) as e:
        # Handle errors by falling back to the device's local time
        #print(f"Could not get time for {timezone.replace('_', ' ')}. Error: {e}")
        month = datetime.now().strftime("%m")
        device_time = datetime.now().strftime(f"Son las %H horas con %M minutos del %d de {months[int(month)-1]} del %Y")
        return device_time

def get_city_date(timezone):
    """ Consigue la fecha del lugar en que se encuentra el usuario dependiendo de la zona horaria, en caso de un error de conección toma la fecha del dispositivo """

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
        #print(f"Could not get time for {timezone.replace('_', ' ')}. Error: {e}")
        month = datetime.now().strftime("%m")
        device_time = datetime.now().strftime(f"Hoy es el %d de {months[int(month)-1]} del %Y")
        return device_time


try:
    # Se consigue la zona horaria y la ciudad del usuario
    ip  = requests.get('https://api.ipify.org').text
    url = f"http://ip-api.com/json/{ip}?lang=es"
    response = requests.get(url)
    data = response.json()
    USER_CITY = data['city']
    USER_TIMEZONE = data['timezone']
except requests.RequestException as e:
    print("Error al obtener la información de ubicación:", e)
    USER_CITY = None

translator = Translator(from_lang='en', to_lang='es')

def get_weather(city):
    """ Consigue el clima actual de la ciudad en la que se encuentra el usuario """
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
    """Reproduce el archivo MP3 descargado usando el reproductor por defecto del sistema """
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
    """ Graba la pantalla y guarda la grabación en el buffer """
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}
        while True:
            start_time = time.time()

            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            with buffer_lock:
                frame_buffer.append(frame)

            time.sleep(max(1.0 / FPS - (time.time() - start_time), 0))

def save_video(filename="clip"):
    """Guarda el buffer a un archivo de video"""
    with buffer_lock:
        frames_to_save = list(frame_buffer)

    if not os.path.exists("./clips"):
        os.makedirs("clips")
    filename = "./clips/"+filename+"-"+time.strftime("%Y%m%d-%H%M%S")+".mp4"
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, FPS, (WIDTH, HEIGHT))

    for frame in frames_to_save:
        out.write(frame)

    out.release()
    print(f"Clip guardado como {filename}")

# Run the screen capture in a separate thread
capture_thread = threading.Thread(target=capture_screen, daemon=True)
capture_thread.start()

# Dataframe de palabras clave
keyword_actions = ["clip", "search","weather", "time","date", "whoami", "music"]
keywords = [
    ["clip", "busca", "clima", "hora", "fecha", "eres", "cancion"],
    [None, "busqu", None, None, "dia", None, "musi"],
    [None, "consulta", None, None, "mes", None, None],
    [None, None, None, None, "ano", None, None],
]
keyword_df = pd.DataFrame(keywords, columns=keyword_actions)

def search_keyword(text):
    """Itera por el dataframe de palabras clave buscando coincidencias"""
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
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    return filename

def transcribe_audio(wav_buffer):
    """Envía el audio a la api de Whisper y retorna el texto"""
    with open(wav_buffer, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def whisper_listen_and_intent(duration):
    """Graba audio por la duración indicada, lo transcribe y busca las palabras clave"""
    audio_chunk = record_audio_chunk(filename="buffer.wav", duration=duration)
    print("Procesando...")
    try:
        text = transcribe_audio(audio_chunk)['text']
        text = unidecode.unidecode(text)
        text = text.lower()

        intent = search_keyword(text)
        if intent is not None:
            act_on_intent(intent)
        else:
            play_audio_file(select_random_file_from_folder(REPEAT_VOICELINES_PATH))
    except Exception as e:
        print(e)
        play_audio_file(select_random_file_from_folder(REPEAT_VOICELINES_PATH))




def whisper_listen_once(duration):
    """Graba audio por la duración indicada, lo transcribe y retorna el texto"""
    audio_chunk = record_audio_chunk(filename="buffer.wav", duration=duration)
    try:
        text = transcribe_audio(audio_chunk)['text']
        print("texto entendido: "+text)
        text = unidecode.unidecode(text)
        text = text.lower()
        return text
    except:
        play_audio_file(select_random_file_from_folder(REPEAT_VOICELINES_PATH))


def act_on_intent(intent):
    """Dependiendo de la palabre clave detectada toma un curso de acción"""
    print("Intención: "+intent)
    if intent == "clip":
        set_talk()
        play_audio_file(select_random_file_from_folder(CLIP_VOICELINES_PATH))
        set_idle()
        save_video()
    elif intent == "search":
        set_talk()
        play_audio_file(select_random_file_from_folder(SEARCH_VOICELINES_PATH))
        set_idle()
        stt = whisper_listen_once(6)
        if stt:
            message = write_mistral(stt)
            print("Respuesta: "+message)
            print("Generando voz...")
            tts(message)
    elif intent == "weather":
        set_talk()
        play_audio_file(select_random_file_from_folder(WAIT_VOICELINES_PATH))
        set_idle()
        tts(get_weather(USER_CITY))
    elif intent == "time":
        set_talk()
        play_audio_file(select_random_file_from_folder(WAIT_VOICELINES_PATH))
        set_idle()
        tts(get_city_time(USER_TIMEZONE))
    elif intent == "date":
        set_talk()
        play_audio_file(select_random_file_from_folder(WAIT_VOICELINES_PATH))
        set_idle()
        tts(get_city_date(USER_TIMEZONE))
    elif intent == "whoami":
        set_talk()
        play_audio_file(select_random_file_from_folder(WHOAMI_VOICELINES_PATH))
        set_idle()
    elif intent == "music":
        set_talk()
        play_audio_file(select_random_file_from_folder(MUSIC_SEARCH_GREET_VOICELINES_PATH))
        set_idle()
        messageMusic = whisper_listen_once(5)  # Obtiene el título o frase para la búsqueda
        if messageMusic:
            set_talk()
            play_audio_file(select_random_file_from_folder(MUSIC_SEARCH_VOICELINES_PATH))
            set_idle()
            download_music(messageMusic)
            play_downloaded_music(messageMusic)

def tts(text):
    """Consigue el audio generado para el texto ingresado"""
    print("Respuesta: "+text)
    body = {"text":text}
    try:
        response = requests.post(TTS_URL,json=body)
        if response.status_code == 200:
            audio_data, samplerate = sf.read(io.BytesIO(response.content), dtype='int16')

            set_talk()
            sd.play(audio_data, samplerate=samplerate)
            sd.wait()
            set_idle()
        else:
            print("Failed to get TTS audio:", response.status_code, response.text)

    except:
        print("Sin conexión a la API")




is_processing_intent = threading.Event()

def process_intent():
    """Function to handle intent processing in a separate thread."""
    set_idle()
    play_audio_file(select_random_file_from_folder(GREET_VOICELINES_PATH))
    whisper_listen_and_intent(3)
    print("Intent processing completed.")

def wakeword_loop():
    """Escucha por las palabras de activación"""
    print("Esperando...")
    predicted = False
    intent_thread = None  # Initialize intent_thread to None

    while True:
        # Capture audio from microphone
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Predict using the wake word model
        prediction = owwModel.predict(audio)
        #print(prediction)

        if prediction['adios'] > SLEEP_THRESHOLD:
            play_audio_file(select_random_file_from_folder(GOODBYE_VOICELINES_PATH))
            delete_music_folder()
            stop_pygame()

            sys.exit()

        if prediction['house'] > WAKEWORD_THRESHOLD and not is_processing_intent.is_set():
            predicted = True
            is_processing_intent.set()

            intent_thread = threading.Thread(target=process_intent)
            intent_thread.start()

        if intent_thread and not intent_thread.is_alive() and predicted:
            is_processing_intent.clear()
            predicted = False
            set_disconnected()
            print("Esperando...")

def play_audio_file(filename,  blocking=True):
    """Reproduce un archivo de audio sin interfáz gráfica"""
    with wave.open(filename, 'rb') as wf:
        audio_data = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    sd.play(audio_array, samplerate=wf.getframerate(), blocking=blocking)

def select_random_file_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return folder_path + (random.choice(files))

if __name__ == "__main__":
    threading.Thread(target=wakeword_loop,daemon=True).start()
    game_loop()
