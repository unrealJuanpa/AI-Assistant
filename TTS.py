import torch
from omegaconf import OmegaConf
import numpy as np
import sounddevice as sd


class ttsModel(object):
    def __init__(self, samplerate:int = 48000, speaker:str = 'es_2', device:str = 'cpu', model_dir:str = 'silero-models-master'):
        self.models = OmegaConf.load('latest_silero_models.yml')

        self.language = 'es'
        self.model_id = 'v3_es'
        self.device = torch.device(device)

        self.model = torch.hub.load(source='local', 
                                    repo_or_dir=model_dir,
                                    model='silero_tts',
                                    language=self.language,
                                    speaker=self.model_id)[0]

        self.model.to(self.device)  # gpu or cpu
        self.speaker = speaker
        self.sample_rate = samplerate


    def __call__(self, text:str):
        #text += '........'
        text += '.....'

        audio = self.model.apply_tts(text=text,
                                     speaker=self.speaker,
                                     sample_rate=self.sample_rate,
                                     put_accent=True,
                                     put_yo=True)
        
        audio = audio.numpy()
        #audio = np.concatenate([audio, np.zeros(shape=int(round(self.sample_rate*0.2)), dtype=np.float32)])
        #sd.play(audio, samplerate=self.sample_rate, blocking=True)        

        return audio
