
from TTS.api import TTS
import sounddevice as sd


tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
speaker_wav = "FNV_MrHouse.wav"  


# Aquí en la ruta se cambia el nombre del archivo en el que se va a guardar el audio
# Me dió pereza poner la ruta entera así que si lo corren directo después se tienen que mover los archivos a la carpeta que corresponda
# No los dejen sueltos porque está hecho para que seleccione audio de una carpeta, no filtra por nombre
def tts(text,n):
    tts_model.tts_to_file(text, speaker_wav=speaker_wav, language='es',file_path=f"house_whoami_{n}.wav") 

n=1

while True:
    text = input("xd")
    print("Iniciando inferencia")
    tts(text,n)
    n += 1