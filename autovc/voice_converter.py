import wandb
from autovc.utils.preprocess_wav import audio_to_melspectrogram
from autovc.speaker_encoder.model import SpeakerEncoder
from autovc.auto_encoder.model_vc import Generator
from autovc.wavernn.model import WaveRNN
import soundfile as sf
import torch
import numpy as np
from autovc.utils.hparams import VoiceConverterParams as hparams


class VoiceConverter:
    """
    Collects Auto Encoder, Speaker Encoder and Vocoder for converting voices.\n
    Can both be used for training a voice converter and using it for converting voices.
    """
    
    def __init__(self, auto_encoder = None, speaker_encoder = None, vocoder = None, **kwargs):  
        
        params = hparams().update(
            AE_params = kwargs.get("auto_encoder_params", {}),
            SE_params = kwargs.get("speaker_encoder_params", {}),
            vocoder_params = kwargs.get("vocoder_params", {})
        )

        self.config = {
            "auto_encoder" : params.AE,
            "speaker_encoder" : params.SE,
            "vocoder" : params.vocoder,
        }

    def convert(self, source, target, outname = "conversion.wav", method = "zero_shot"):
        """
        Gives the features of the target to the content of the source

        Parameters
        ---------
        source:
            soundfile with content to convert
        target:
            soundfile containing the voice of the person which features to use
            if proper training has been done, it can also be a string with the name of the person
        outname:
            filename for converted sound
        method:
            how to convert the voice, can be one of ['zero_shot', 'one_to_one']
        """
        pass

    def train(self, model_type = "auto_encoder"):
        """
        Trains a model

        Parameters
        ---------
        model_type:
            which model type to train, can be one of ['auto_encoder', 'speaker_encoder']
        """
        pass

    def load(self, path):
        pass

    def convert_multiple(self):
        """
        Uses the convert function on multiple files
        """
        pass


if __name__ == "__main__":
    vc = VoiceConverter()
    print(vc.config)