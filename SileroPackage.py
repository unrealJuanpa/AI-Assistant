
#from STT import sttModel
from TTS import ttsModel

#model1 = sttModel()
model2 = ttsModel()

#text = model1('speech_orig_es.wav')
text = 'hola mundo, esta es una prueba de generacion de voz'
print(text)
model2(text)
