from logging import setLogRecordFactory
from re import S
import wandb
from wandb.sdk.wandb_run import Run
# from autovc.utils.audio import audio_to_melspectrogram, preprocess_wav, remove_noise
from autovc.utils.core import retrieve_file_paths
import soundfile as sf
import torch
import numpy as np
# from autovc.utils.hparams import VoiceConverterParams
from autovc.utils.dataloader import TrainDataLoader, SpeakerEncoderDataLoader
from autovc.utils.model_loader import load_models
import time
import os
from itertools import product
from autovc.utils.hparams import AutoEncoderParams, SpeakerEncoderParams, WaveRNNParams, VoiceConverterParams, Namespace
from autovc.audio import Audio, spectrogram


class VoiceConverter:
    """
    Collects a pretrained Auto Encoder, Speaker Encoder and Vocoder for converting voices.\n
    Can both be used for continuing training of a voice converter and for converting voices.
    All default values can be found in `autovc/utils/hparams.py`
    """
    
    def __init__(self,
        auto_encoder = None, 
        speaker_encoder = None, 
        vocoder = None, 
        verbose = True,
        **kwargs 
    ):
        """
        Initialises a VoiceConverter instance

        Params
        ------
        auto_encoder:
            path to an auto encoder model
        speaker_encoder:
            path to a speaker encoder model
        vocoder:
            path to a vocoder model (for now only WaveRNN models are supported)
        
        Kwargs
        ------
        auto_encoder_params:
            a dictionary with params for the auto encoder, default values can be found in `autovc/utils/hparams.py`
        speaker_encoder_params:
            a dictionary with params for the speaker encoder, default values can be found in `autovc/utils/hparams.py`
        vocoder_params:
            a dictionary with params for the vocoder, default values can be found in `autovc/utils/hparams.py`
        wandb_params:
            a dictionary with parameters to pass to `wandb.init()` when creating a run
        device:
            which device code should be run on, defaults to "cuda" if available else "cpu"
        """
       
        # default wandb behaviour
        self.wandb_run = None
        wandb_defaults = {
            # "sync_tensorboard":True, 
            "reinit":True,
            "entity" : "deep_voice_inc",
            # "name" : self.run_name,
            "project" : "GettingStarted", # wandb project name, each project correpsonds to an experiment
            # "dir" : "logs/" + "GetStarted", # dir to store the run in
            # "group" : self.agent_name, # uses the name of the agent class
            "save_code":True,
            "mode" : "online",
        }
        wandb_defaults.update(kwargs.pop("wandb_params", {}))

        # setup config with params
        self.verbose = verbose
        self.config = {
            "AE_params" : AutoEncoderParams().update(kwargs.pop("auto_encoder_params", {})).get_collection(),
            "SE_params" : SpeakerEncoderParams().update(kwargs.pop("speaker_encoder_params", {})).get_collection(),
            "vocoder_params" : WaveRNNParams().update(kwargs.pop("vocoder_params", {})).get_collection(),
            "wandb_params" : wandb_defaults,
            **VoiceConverterParams().update(kwargs, auto_encoder, speaker_encoder, vocoder).get_collection(),
        }

        # initialise models
        self._init_models()
        
        

    def convert(self, source, target,  sr = 22050, out_name = None, out_dir = None, pipes = {}, pipe_args = {}):
        """
        Gives the features of the target to the content of the source

        Parameters
        ---------
        source:
            soundfile with content to convert
        target:
            soundfile containing the voice of the person which features to use
            if proper training has been done, it can also be a string with the name of the person
        sr:
            The sample rate to use for the audio
        outname:
            filename for converted sound
        out_dir:
            the folder/directory to store the converted file in, if this is 'wandb' the converted audio will be logged to this run
            all conversions not stored in WANDB will be saved in `results/` folder
        pipes:
            Dictionary with pipelines for preprocessing audio, keys must be one of ['target', 'source', 'output'] 
            E.g. {"source" : ["normalize_volume", "remove_noise"], "target" : ["normalize_volume", "remove_noise"], "output" : ["remove_noise"] 
        pipe_args:
            Dictionary of dictionaries where outer key matches the pipe type and inner key matches string given in pipe and values are arguments to give to the function. 
            E.g. if "normalize_volume" is given in target pipe, pipe_args = {"target" : {"normalize_volume" : "target_dBFS" = -30}} can be given to select the normalisation volume
        """

        print("Beginning conversion...")
        # source_wav, target_wav = preprocess_wav(source), preprocess_wav(target)
        audio_src = Audio(source, sr).preprocess("convert_in", *pipes.get("source", []), **pipe_args.get("source", {}))
        

        if not target in self.SE.speakers.keys():
            audio_trg = Audio(target, sr).preprocess("convert_in", *pipes.get("target", []), **pipe_args.get("target", {}))
            c_target = self.SE.embed_utterance(audio_trg.wav).unsqueeze(0)
        else:
            c_target = self.SE.speakers[target].unsqueeze(0)

       
       
        # Generate speaker embeddings
        c_source = self.SE.embed_utterance(audio_src.wav).unsqueeze(0)
        # c_target = self.SE.embed_utterance(audio_trg.wav).unsqueeze(0)

        # Create mel spectrogram
        # mel_spec = torch.from_numpy(audio_to_melspectrogram(audio_src.wav)).unsqueeze(0)
        mel_spec = spectrogram.mel_spectrogram(audio_src.wav, "auto_encoder").unsqueeze(0)
        
        # Convert
        out, post_out, content_codes = self.AE(mel_spec, c_source, c_target)

        # Use the Vocoder to generate waveform (use post_out as input)
        waveform = self.vocoder.generate(post_out)

        # # reduce noise
        # if clean:
        #     waveform = remove_noise(waveform, self.config["vocoder_params"].get("sample_rate"))

        # ensure a np array is returned
        waveform = np.asarray(waveform)
        audio_out = Audio(waveform, sr = sr, sr_org = self.config["vocoder_params"].get("sample_rate"))
        audio_out.preprocess("convert_out", *pipes.get("output", []), **pipe_args.get("output", {}))


        # Generate .wav file frowm waveform
        if out_name is None:
            src_speaker = os.path.splitext(os.path.basename(source))[0]
            trg_speaker = os.path.splitext(os.path.basename(target))[0]
            out_name = f"{src_speaker}_to_{trg_speaker}.wav"
        
        # if out_dir == self.wandb_run and self.wandb_run is not None:
        if out_dir == "wandb":
            # TODO log as table
            assert self.wandb_run is not None, "A wandb run has to be setup with _init_wandb() to save conversion in wandb"
            self.wandb_run.log({out_name.replace(".wav", "") : wandb.Audio(audio_out.wav, caption = out_name, sample_rate = audio_out.sr)})
        else:
            if out_dir is not None:
                out_dir = out_dir if out_dir.startswith("results") else "results/" + out_dir 
                out_name = out_dir.strip("/") + "/" + out_name
                os.makedirs(out_dir, exist_ok=True) # create folder
            else:
                os.makedirs("results", exist_ok=True) # create folder
                out_name = out_name if out_name.startswith("results") else "results/" + out_name 
    
            sf.write(out_name, audio_out.wav, samplerate =audio_out.sr)
    
        return audio_out.wav, audio_out.sr

    def train(self, model_type = "auto_encoder", n_epochs = 2, conversion_examples = None, data_path = 'data/samples', **train_params):
        """
        Trains a model

        Parameters
        ---------
        model_type:
            which model type to train, can be one of ['auto_encoder', 'speaker_encoder']
        conversion_examples:

        """
        # asssert valid model type
        if model_type not in ["auto_encoder", "speaker_encoder"]:
            raise ValueError(f"'{model_type}' is not a supported model_type")

        # update config with train params
        model_params = "AE_params" if model_type == "auto_encoder" else "SE_params"
        self.config[model_params].update({**train_params, "n_epochs" : n_epochs, "data_path" : data_path})

        # create a wandb run
        self._init_wandb()

        start_time = time.time()
        # print(f"Starting to train {model_type}...")
        if model_type == "auto_encoder":
            # print("Training device: ", self.AE.params.device)
            params = AutoEncoderParams().update(self.config[model_params])
            dataset = TrainDataLoader(**params.get_collection("dataset"), speaker_encoder = self.SE)
            # dataset = TrainDataLoader(speaker_encoder = self.SE, chop = True, data_path = 'data/test_data')
            dataloader = dataset.get_dataloader(**params.get_collection("dataloader"))
            self.AE.learn(dataloader, wandb_run = self.wandb_run, **params.get_collection())
        elif model_type == "speaker_encoder":
            datadir = {'hilde': ['data/test_data/hilde_7sek'], 'hague': ['data/test_data/HaegueYang_10sek']}
            dataset = SpeakerEncoderDataLoader(datadir, device = self.config.get("device", 'cuda'))
            dataloader = dataset.get_dataloader(batch_size = 1024)
            self.SE.learn(dataloader, n_epochs = 128, wandb_run = self.wandb_run,  log_freq = 16, save_freq = 32)
            # raise NotImplementedError()
            
        
        print(f"Training finished in {time.time() - start_time}")

        # conversion example
        if conversion_examples is None:
            self.convert(
                # "data/samples/mette_183.wav", 
                # "data/samples/chooped7.wav",
                "data/samples/hilde_1.wav", 
                "data/samples/HaegueYang_5.wav",
                out_dir = "wandb" if not self.wandb_run.mode == "disabled" else ".",
                # out_folder=os.path.join(self.wandb_run.dir, "conversions")
            )
            # self.wandb_run.log({"example" : wandb.Audio(wav, caption = "test", sample_rate = sr)})
        elif conversion_examples:
            self.convert_multiple(
                conversion_examples[0], 
                conversion_examples[1], 
                out_dir = "wandb" if not self.wandb_run.mode == "disabled" else ".",
            )



    def convert_multiple(self, sources, targets, match_method = "all_combinations", bidirectional = False, **convert_params):
        """
        Uses the convert function on multiple files

        Params
        ------
        sources:
            path or list of paths to source files
        targets:
            path or list of paths to target files
        method:
            tells how to match sources and targets
            'all_combinations' will match each source with each target
            'align' will match sources and targets with same index in the given list
        **convert_params
            params to give to VoiceConverter.convert
        """

        sources = retrieve_file_paths(sources)
        if len(targets) > 1 or  os.path.isdir(targets[0]):
            # targets = targets[0]
        # else:
            targets = retrieve_file_paths(targets)
        wavs, sample_rates = [], []

        if match_method == "align":
            assert len(sources) == len(targets)
            matches = [*zip(sources, targets)]
        elif match_method == "all_combinations":
            matches = product(sources, targets)

        for source, target in matches:
            wav, sr = self.convert(source, target, **convert_params)
            wavs.append(wav)
            sample_rates.append(sr)
        
        if bidirectional:
            wav, sr = self.convert_multiple(target, sources, match_method, **convert_params)
            wavs.extend(wav)
            sample_rates.extend(sr)

        return wavs, sample_rates

    
    def _init_wandb(self, **params):
        # skip if setup has already been done
        if self.wandb_run is not None:
            return None

        # overwrite defaults with parsed arguments
        self.config["wandb_params"].update(params)
        self.config["wandb_params"].update({"dir" : "logs/" + self.config["wandb_params"].get("project")})

        # create directory for logs if first run in project
        if not os.path.exists(self.config["wandb_params"].get("dir", "logs")+"/wandb"): 
            os.makedirs(self.config["wandb_params"].get("dir", "logs")+"/wandb") 

        # init wandb       
        self.wandb_run = wandb.init(**self.config["wandb_params"], config = self.config)
        self.config = self.wandb_run.config

    
    def close(self):
        """
        Closes the necesary processes
        """
        if self.wandb_run is not None:
            self.wandb_run.finish()
        wandb.finish()

    def _init_models(self):
        type_to_dir = {"auto_encoder" : "AutoVC", "speaker_encoder" : "SpeakerEncoder", "vocoder" : "WaveRNN"}
        type_to_artifact = {"auto_encoder" : "AutoEncoder", "speaker_encoder" : "SpeakerEncoder"}
        model_paths= [self.config.get("AE_model"), self.config.get("SE_model"), self.config.get("vocoder_model")]
        model_types = list(type_to_dir.keys())

        for i in range(len(model_paths)):
            if not os.path.isfile(model_paths[i]):
                if self.config["wandb_params"].get("mode") in ["disabled", "offline"]:
                    raise ValueError(f"Model {model_paths[i]} was not found locally and it was not possible to look in wandb, as the wandb mode was not set to 'online'")
                print(f"Looking for model {model_paths[i]} in wandb...")
                self._init_wandb()
                artifact = self.wandb_run.use_artifact(model_paths[i], type=type_to_artifact[model_types[i]])
                model_path = "models/" + type_to_dir[model_types[i]]+"/"+os.path.basename(artifact.file())
                if os.path.isfile(model_path):
                    raise ValueError(f"The model '{model_path}' already exists, please use this path instead or delete the file if you want to use the model from wandb")
    
                artifact.download("models/" + type_to_dir[model_types[i]])
                model_paths[i] = model_path
        
        self.AE, self.SE, self.vocoder = load_models(
            model_types= model_types,
            model_paths= model_paths,
            params = [self.config["AE_params"], self.config["SE_params"], self.config["vocoder_params"]],
            device = self.config.get("device", None),
        )


if __name__ == "__main__":
    from autovc.utils.argparser import parse_vc_args
    # args = "-mode train -model_type auto_encoder -wandb_params mode=online -n_epochs 1 -data_path data/samples -data_path_excluded data/samples/chooped7.wav -auto_encoder deep_voice_inc/SpeakerEncoder/model_20220113.pt:v4"
    # args = "-mode convert -sources data/samples/hilde_301.wav -targets data/samples/chooped7.wav -convert_params pipes={output:[normalize_volume,remove_noise],source:[normalize_volume]} -auto_encoder models/AutoVC/AutoVC_SMK.pt"# -auto_encoder deep_voice_inc/AutoEncoder/model_20220114.pt:v0"
    # args = "-mode train -model_type auto_encoder -wandb_params mode=online project=SpeakerEncoder2 -data_path data/test_data -n_epochs 16 -speaker_encoder SpeakerEncoder3.pt -auto_encoder_params batch_size=32 chop=True speakers=True save_freq=64 freq=80 "
    # args = "-mode train -model_type speaker_encoder -wandb_params mode=online project=SpeakerEncoder2 -data_path data/test_data -n_epochs 128 -speaker_encoder_params SE_model=models/SpeakerEncoder/SpeakerEncoder.pt"
    # args = "-mode convert -sources data/samples/mette_183.wav -targets data/samples/chooped7.wav"
    args = None # make sure this is used when not testing
    args = vars(parse_vc_args(args))

    vc = VoiceConverter(
        auto_encoder=args.pop("auto_encoder", None),
        speaker_encoder= args.pop("speaker_encoder", None),
        vocoder= args.pop("vocoder", None),
        device = args.pop("device", None),
        verbose=args.pop("verbose", None),
        auto_encoder_params = args.pop("auto_encoder_params", {}),
        speaker_encoder_params = args.pop("speaker_encoder_params", {}),
        vocoder_params = args.pop("vocoder_params", {}),
        wandb_params = args.pop("wandb_params", {})
    )

    # get mode
    mode = args.pop("mode")

    if mode == "train":
        assert all([args.__contains__(key) for key in ["model_type", "n_epochs"]])
        convert_examples = all([args.__contains__(key) for key in ["conversions_sources", "conversions_targets"]])
        # train
        vc.train(
            **args
            # model_type = args.model_type, 
            # n_epochs = args.n_epochs, 
            # data_path=args.data_path,
            # conversion_examples= None if not convert_examples else [args.conversion_sources, args.conversion_targets]
        )
    elif mode == "convert":
        assert all([args.__contains__(key) for key in ["sources", "targets"]])
        vc.convert_multiple(
            **args.pop("convert_params", {}),
            **args
            # sources = args.conversion_sources, 
            # targets = args.conversion_targets,
            # save_folder = args.results_dir,
            # bidirectional = args.bidirectional,
            # method = args.conversion_data_alignment

        )

    
    vc.close()















    # vc = VoiceConverter(wandb_params = {"mode" : "online", "project": "test21"})#, auto_encoder="models/AutoVC/AutoVC_SMK_20211104_original_step42.05k.pt")
    # print(vc.config)
    # vc.train("auto_encoder", conversion_examples=[["data/samples/mette_183.wav", 
                # "data/samples/chooped7.wav"], "data/samples/chooped7.wav"])

    # vc.setup_wandb_run()
    # vc.convert("data/samples/mette_183.wav", "data/samples/chooped7.wav")

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


    # vc.close()