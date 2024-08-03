import torch
import os
from glob import glob


class sttModel(object):
    def __init__(self, device:str = 'cpu', model_dir:str = 'silero-models-master'):
        self.device = torch.device(device)

        self.model, self.decoder, self.utils = torch.hub.load(source='local', 
                                                              repo_or_dir=model_dir,
                                                              model='silero_stt',
                                                              language='es',
                                                              device=self.device)

        (self.read_batch, self.split_into_batches, self.read_audio, self.prepare_model_input) = self.utils

    def __call__(self, wavfile:str):
        outstr = ''
        test_files = glob(wavfile)
        batches = self.split_into_batches(test_files, batch_size=10)
        input = self.prepare_model_input( self.read_batch(batches[0]), device=self.device )

        output = self.model(input)
        for example in output:
            outstr += self.decoder(example.cpu())

        return outstr


