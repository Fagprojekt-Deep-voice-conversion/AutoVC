from logging import setLogRecordFactory
import wandb
from wandb.sdk.wandb_run import Run
from autovc.utils.audio import audio_to_melspectrogram, remove_noise
from autovc.utils.core import retrieve_file_paths
import soundfile as sf
import torch
import numpy as np
# from autovc.utils.hparams import VoiceConverterParams
from autovc.utils.dataloader import TrainDataLoader
from autovc.utils.model_loader import load_models
import time
import os
from itertools import product
from autovc.utils.hparams import AutoEncoderParams, SpeakerEncoderParams, WaveRNNParams, VoiceConverterParams, Namespace


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
       
        # default wandb behaviour
        self.wandb_run = None
        wandb_defaults = {
            # "sync_tensorboard":True, 
            "reinit":True,
            "entity" : "deep_voice_inc",
            # "name" : self.run_name,
            "project" : "GetStarted", # wandb project name, each project correpsonds to an experiment
            "dir" : "logs/" + "GetStarted", # dir to store the run in
            # "group" : self.agent_name, # uses the name of the agent class
            "save_code":True,
            "mode" : "online",
        }

        # setup config with params
        self.verbose = verbose
        self.config = {
            "AE_params" : AutoEncoderParams().update(kwargs.pop("auto_encoder_params", {})).get_collection(),
            "SE_params" : SpeakerEncoderParams().update(kwargs.pop("speaker_encoder_params", {})).get_collection(),
            "vocoder_params" : WaveRNNParams().update(kwargs.pop("vocoder_params", {})).get_collection(),
            "wandb_params" : wandb_defaults.update(kwargs.pop("wandb_params", {})),
            **VoiceConverterParams().update(kwargs).get_collection(),
        }

        # initialise models
        
        self.AE, self.SE, self.vococder = load_models(
            model_types= ["auto_encoder", "speaker_encoder", "vocoder"],
            model_paths= [
                # 'models/AutoVC/AutoVC_SMK.pt' if auto_encoder is None else auto_encoder, 
                'models/AutoVC/AutoVC_seed40_200k.pt' if auto_encoder is None else auto_encoder, 
                'models/SpeakerEncoder/SpeakerEncoder.pt' if speaker_encoder is None else speaker_encoder,
                'models/WaveRNN/WaveRNN_Pretrained.pyt' if vocoder is None else vocoder
                ],
            device = self.config.get("device", None)
        )
        
        

    def convert(self, source, target, out_name = None, out_folder = None, method = "zero_shot", wandb_run = None):
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
        
        # Convert
        out, post_out, content_codes = self.AE(mel_spec, c_source, c_target)

        # Use the Vocoder to generate waveform (use post_out as input)
        waveform = self.vococder.generate(post_out)

        # reduce noise
        waveform = remove_noise(waveform, self.vococder.params.sample_rate)

        # ensure a np array is returned
        waveform = np.asarray(waveform)

        # Generate .wav file frowm waveform
        if out_name is None:
            src_speaker = os.path.splitext(os.path.basename(source))[0]
            trg_speaker = os.path.splitext(os.path.basename(target))[0]
            out_name = f"{src_speaker}_to_{trg_speaker}.wav"
        
        if out_folder == self.wandb_run and self.wandb_run is not None:
            # TODO log as table
            out_folder.log({out_name.replace(".wav", "") : wandb.Audio(waveform, caption = out_name, sample_rate = self.vococder.params.sample_rate)})
        else:
            if out_folder is not None:
                out_name = out_folder.strip("/") + "/" + out_name
                os.makedirs(out_folder, exist_ok=True) # create folder
    
            sf.write(out_name, waveform, samplerate =self.vococder.params.sample_rate)
    
        return waveform, self.vococder.params.sample_rate

    def train(self, model_type = "auto_encoder", n_epochs = 2, conversion_examples = None, data = 'data/samples', **train_params):
        """
        Trains a model

        Parameters
        ---------
        model_type:
            which model type to train, can be one of ['auto_encoder', 'speaker_encoder']
        conversion_examples:

        """
        # create a wandb run
        if self.wandb_run is None:
            self.setup_wandb_run(**self.config["wandb_params"])

        start_time = time.time()
        print(f"Starting to train {model_type}...")
        if model_type == "auto_encoder":
            print(self.AE.params.device)
            dataset = TrainDataLoader(data_dir_path = data, speaker_encoder = self.SE)
            dataloader = dataset.get_dataloader(batch_size = 2, shuffle = True)
            self.AE.learn(dataloader, n_epochs = n_epochs, wandb_run = self.wandb_run, **train_params)
        elif model_type == "speaker_encoder":
            raise NotImplementedError()
        else:
            raise ValueError(f"'{model_type}' is not a supported model_type")
        
        print(f"Training finished in {time.time() - start_time}")

        # conversion example
        if conversion_examples is None:
            self.convert(
                # "data/samples/mette_183.wav", 
                # "data/samples/chooped7.wav",
                "data/samples/hilde_301.wav", 
                "data/samples/HaegueYang_5.wav",
                out_folder = self.wandb_run,
                # out_folder=os.path.join(self.wandb_run.dir, "conversions")
            )
            # self.wandb_run.log({"example" : wandb.Audio(wav, caption = "test", sample_rate = sr)})
        elif conversion_examples:
            self.convert_multiple(conversion_examples[0], conversion_examples[1], save_folder = self.wandb_run)



    def convert_multiple(self, sources, targets, save_folder = None, method = "all_combinations", bidirectional = False):
        """
        Uses the convert function on multiple files

        Params
        ------
        sources:
            path or list of paths to source files
        targets:
            path or list of paths to target files
        save_folder:
            folder to save converted files in
        method:
            tells how to match sources and targets
            'all_combinations' will match each source with each target
            'align' will match sources and targets with same index in the given list
        """

        sources = retrieve_file_paths(sources)
        targets = retrieve_file_paths(targets)
        wavs, sample_rates = [], []

        if method == "align":
            assert len(sources) == len(targets)
            matches = [*zip(sources, targets)]
        elif method == "all_combinations":
            matches = product(sources, targets)

        for source, target in matches:
            wav, sr = self.convert(source, target, out_folder = save_folder)
            wavs.append(wav)
            sample_rates.append(sr)
        
        if bidirectional:
            wav, sr = self.convert_multiple(target, sources, save_folder, method)
            wavs.extend(wav)
            sample_rates.extend(sr)

        return wavs, sample_rates

    
    def setup_wandb_run(self, **params):
        

        # self.wandb_run = wandb.init(entity="deep_voice_inc", project= "GetStarted", config = self.config)

         # overwrite defaults with parsed arguments
        # wand_args = {**wand_defaults, **params}
        self.config["wandb_params"].update(params)

        # create directory for logs if first run in project
        if not os.path.exists(self.config["wandb_params"].get("dir", "logs")+"/wandb"): 
            os.makedirs(self.config["wandb_params"].get("dir", "logs")+"/wandb") 

        # init wandb       
        self.wandb_run = wandb.init(**self.config["wandb_params"], config = self.config)
        self.config = self.wandb_run.config

        # watch models
        self.wandb_run.watch(self.AE, log_freq = self.AE.params.log_freq)
        self.wandb_run.watch(self.SE, log_freq = self.SE.params.log_freq)


if __name__ == "__main__":
    vc = VoiceConverter(wandb_params = {"mode" : "online"})#, auto_encoder="models/AutoVC/AutoVC_SMK_20211104_original_step42.05k.pt")
    # print(vc.config)
    # vc.train("auto_encoder", conversion_examples=[["data/samples/mette_183.wav", 
                # "data/samples/chooped7.wav"], "data/samples/chooped7.wav"])

    vc.setup_wandb_run()
    vc.convert("data/samples/mette_183.wav", "data/samples/chooped7.wav")

    # vc.convert_multiple(
    #     ["data/samples/hillary_116.wav", "data/samples/mette_183.wav"], 
    #     ["data/samples/mette_183.wav", "data/samples/hillary_116.wav"],
    #     method = "all_combinations",
    #     save_folder = "test"    
    # )

    # vc.train(data = "data/SMK_train/newest_trial", n_epochs = 1)
    # vc.setup_wandb_run(name = "SMK")
    # vc.train(data = "data/SMK_train/20211104", n_epochs = 1)
    # vc.train(data = "data/SMK_train", n_epochs = 1)
    # vc.train(data = "data/samples", n_epochs = 1)
