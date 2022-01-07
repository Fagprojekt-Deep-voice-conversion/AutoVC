from librosa.filters import mel
import wandb
from autovc.utils.audio import audio_to_melspectrogram, remove_noise
from autovc.speaker_encoder.model import SpeakerEncoder
from autovc.auto_encoder.model_vc import Generator
from autovc.wavernn.model import WaveRNN
import soundfile as sf
import torch
import numpy as np
# from autovc.utils.hparams import VoiceConverterParams
from autovc.utils.dataloader import TrainDataLoader
from autovc.utils.model_loader import load_models
import time


class VoiceConverter:
    """
    Collects Auto Encoder, Speaker Encoder and Vocoder for converting voices.\n
    Can both be used for training a voice converter and using it for converting voices.
    """
    
    def __init__(self,
        auto_encoder = None, 
        speaker_encoder = None, 
        vocoder = None, 
        verbose = True,
        **kwargs 
    ):
    
    
     
        
        # params = VoiceConverterParams().update(
        #     AE_params = kwargs.pop("auto_encoder_params", {}),
        #     SE_params = kwargs.pop("speaker_encoder_params", {}),
        #     vocoder_params = kwargs.pop("vocoder_params", {})
        #     **kwargs
        # )

        # setup config with params
        self.verbose = verbose
        self.config = {
            "AE_params" : kwargs.pop("auto_encoder_params", {}),
            "SE_params" : kwargs.pop("speaker_encoder_params", {}),
            "vocoder_params" : kwargs.pop("vocoder_params", {}),
            **kwargs
        }

        # initialise models
        self.AE, self.SE, self.vococder = load_models(
            model_types= ["auto_encoder", "speaker_encoder", "vocoder"],
            model_paths= [
                'models/AutoVC/AutoVC_SMK.pt' if auto_encoder is None else auto_encoder, 
                'models/SpeakerEncoder/SpeakerEncoder.pt' if speaker_encoder is None else speaker_encoder,
                'models/WaveRNN/WaveRNN_Pretrained.pyt' if vocoder is None else vocoder
                ],
            device = self.config.get("device", None)
        )

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

        print("Beginning conversion...")

        # Generate speaker embeddings
        c_source = self.SE.embed_utterance(source).unsqueeze(0)
        c_target = self.SE.embed_utterance(target).unsqueeze(0)

        # Create mel spectrogram
        mel_spec = torch.from_numpy(audio_to_melspectrogram(source)).unsqueeze(0)
        # mel_spec = mel_spec
        # Convert:
        #   out is the converted output
        #   post_out is the refined (by postnet) out put
        #   content_codes is the content vector - the content encoder output
        out, post_out, content_codes = self.AE(mel_spec, c_source, c_target)

        # Use the Vocoder to generate waveform (use post_out as input)
        waveform = self.vococder.generate(post_out)

        # reduce noise
        waveform = remove_noise(waveform, self.vococder.params.sample_rate)

        # Generate .wav file frowm waveform
        sf.write(outname, np.asarray(waveform), samplerate =self.vococder.params.sample_rate)


    
    # remember to call model.eval()

    def train(self, model_type = "auto_encoder"):
        """
        Trains a model

        Parameters
        ---------
        model_type:
            which model type to train, can be one of ['auto_encoder', 'speaker_encoder']
        """
    #  remember to call model.train()

        start_time = time.time()
        print(f"Starting to train {model_type}...")
        if model_type == "auto_encoder":
            dataset = TrainDataLoader(data_dir_path = 'data/samples', speaker_encoder = self.SE)
            dataloader = dataset.get_dataloader(batch_size = 2, shuffle = True)
            self.AE.learn(dataloader, n_epochs = 2)
        elif model_type == "speaker_encoder":
            raise NotImplementedError()
        else:
            raise ValueError(f"'{model_type}' is not a valid model_type")
        
        print(f"Training finished in {time.time() - start_time}")

    # def load_new_model(self, path):
    #     pass

    def convert_multiple(self):
        """
        Uses the convert function on multiple files
        """
        pass


if __name__ == "__main__":
    vc = VoiceConverter()
    # print(vc.config)
    # vc.train("auto_encoder")
    vc.convert("data/samples/mette_183.wav", "data/samples/chooped7.wav")