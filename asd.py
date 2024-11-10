import requests
import sounddevice as sd
import io
import soundfile as sf

body = {"text":"Momazos diego"}
response = requests.post("http://191.239.119.23:5000/tts",json=body)

if response.status_code == 200:
    # Load audio data from response content
    audio_data, samplerate = sf.read(io.BytesIO(response.content), dtype='int16')

    # Play the audio
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()
else:
    print("Failed to get TTS audio:", response.status_code, response.text)
