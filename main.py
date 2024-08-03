import ollama
from system_prompt import personality
from TTS import ttsModel
import sounddevice as sd
import threading
import time
import keyboard
import numpy as np
import whisper


model = whisper.load_model("base", device='cuda')
stopSeq = '?!.,):;\n-'

ttsStringQueue = []
ttsAudioQueue = []
ttsEngine = ttsModel(samplerate=24000, device='cuda') # 8000, 24000


def calculateTTSService():
    global ttsEngine
    while True:
        try:
            if len(ttsStringQueue) > 0:
                ttsAudioQueue.append(ttsEngine(ttsStringQueue.pop(0)))
            else: 
                time.sleep(0.5)
        except Exception as e:
            print(str(e))
        

def sayTTSService():
    while True:
        if len(ttsAudioQueue) > 0:
            sd.play(ttsAudioQueue.pop(0), samplerate=ttsEngine.sample_rate, blocking=True)
        else: 
            time.sleep(0.5)



calculateThread = threading.Thread(target=calculateTTSService)
calculateThread.start()

sayThread = threading.Thread(target=sayTTSService)
sayThread.start()


history = [{'role': 'system', 'content': personality}]
tms = 0.5
whisper_samplerate = int(16000)
audio = np.zeros(shape=(int(30*whisper_samplerate),1), dtype=np.float32)


while True:

    try:
        print('\nEsperando pulsacion')

        while not keyboard.is_pressed('.'):
            time.sleep(tms)

        c = 0
        print('Escuchando...')
        sd.rec(samplerate=whisper_samplerate, channels=1, out=audio, blocking=False, dtype=np.float32)

        while keyboard.is_pressed('.'):
            time.sleep(tms)
            c += 1
            print(c)

        sd.stop()

        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio[0: int(whisper_samplerate*c*tms), 0])).to(model.device)
        options = whisper.DecodingOptions(task='transcribe', language="spanish", fp16=False)
        result = whisper.decode(model, mel, options)

        print('Recognized text:')
        print(result.text)


        history.append({'role': 'user', 'content': result.text})

        stream = ollama.chat(
            model='llama3.1',
            messages=history,
            stream=True,
        )

        msg = ''
        l = ''

        for chunk in stream:
            token = chunk['message']['content']
            print(token, end='', flush=True)
            msg = msg + token
            l = l + token

            if token in stopSeq: 
                ttsStringQueue.append(l)
                l = ''

        history.append({'role': 'assistant', 'content': msg})
    except Exception as e:
        print(str(e))
        exit()
