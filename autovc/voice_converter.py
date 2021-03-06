from genericpath import exists
from torchaudio import datasets
import wandb
from autovc.utils import retrieve_file_paths, pformat
import soundfile as sf
import torch
import numpy as np
from autovc.utils.dataloader import AutoEncoderDataset, SpeakerEncoderDataset
from autovc.models import load_models
import time
import os
from itertools import product
from autovc.utils.hparams import *
from autovc.audio import Audio, spectrogram
import torch


class VoiceConverter:
    """
    Collects a pretrained Auto Encoder, Speaker Encoder and Vocoder for converting voices.\n
    Can both be used for continuing training of a voice converter and for converting voices.
    All default values can be found in `autovc/utils/hparams.py`
    """
    
    def __init__(self,
        auto_encoder = VoiceConverterParams["default_models"]["auto_encoder"], 
        auto_encoder_params = {},
        speaker_encoder = VoiceConverterParams["default_models"]["speaker_encoder"], 
        speaker_encoder_params = {},
        vocoder = VoiceConverterParams["default_models"]["vocoder"], 
        vocoder_params = {},
        wandb_params = {},
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose = True,
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
       
        # set values
        self.wandb_run = None
        self.verbose = verbose

        # update params and add to config
        AutoEncoderParams.update(auto_encoder_params)
        SpeakerEncoderParams.update(speaker_encoder_params)
        WaveRNNParams.update(vocoder_params)
        WandbParams.update(wandb_params)
        
        self.config = {
            "model_names" : {"auto_encoder" : auto_encoder, "speaker_encoder" : speaker_encoder, "vocoder" : vocoder},
            "auto_encoder" : AutoEncoderParams,
            "speaker_encoder" : SpeakerEncoderParams,
            "vocoder" : WaveRNNParams,
            "wandb" : WandbParams,
            "device" : torch.device(device) if isinstance(device, str) else device,
        }

        # initialise models
        self.__init_models()

        # add known speakers to config
        self.config["speaker_encoder"]["mean_speakers"] = {key : "learned_previously" for key in self.SE.speakers.keys()}        
        

    def convert(self, 
        source, 
        target,  
        sr = VoiceConverterParams["convert"]["sr"], 
        save_name = VoiceConverterParams["convert"]["save_name"], 
        save_dir = VoiceConverterParams["convert"]["save_dir"], 
        preprocess = VoiceConverterParams["convert"]["preprocess"],
        preprocess_args = VoiceConverterParams["convert"]["preprocess_args"],
        outprocess = VoiceConverterParams["convert"]["outprocess"],
        outprocess_args = VoiceConverterParams["convert"]["outprocess_args"],
        audio_log_dict = {},
        device = 'cuda',
        **kwargs
    ):
        """
        Gives the features of the target to the content of the source

        Parameters
        ---------
        source:
            soundfile with content to convert
        target:
            soundfile containing the voice of the person which features to use
            If the speaker encoder is trained for a specific person, a string with this key can be passed instead.
        sr:
            The sample rate to use for the audio
        save_name:
            Filename for converted sound. 
            If False, no file is saved
        save_dir:
            the folder/directory to store the converted file in, if this is 'wandb' the converted audio will be logged to this run
            all conversions not stored in WANDB will be saved in `results/` folder
        preprocess:
            Pipeline to use for preprocessing (see `autovc.audio.Audio.preprocess`)
        preprocess_args:
            Pipeline args to use for preprocessing (see `autovc.audio.Audio.preprocess`)
        outprocess:
            Pipeline to use for outprocessing (see `autovc.audio.Audio.preprocess`)
        outprocess_args:
            Pipeline args to use for outprocessing (see `autovc.audio.Audio.preprocess`)
        audio_log_dict:
            Values to log along with the converted audio in wandb. this requires save_dir to be 'wandb'
        **kwargs:
            Kwargs are given to `mel_spec_auto_encoder`
        Return
        ------
        audio_out:
            An Audio object (see 'autovc/audio/__init__.py') with the converted voice
        """

        print(f"Beginning conversion of '{source}' to '{target}'...")

        # source_wav, target_wav = preprocess_wav(source), preprocess_wav(target)

        # source data
        audio_src = Audio(source, sr)
        audio_src.preprocess(*preprocess, **preprocess_args)
        # c_source = self.SE.embed_utterance(audio_src.wav).unsqueeze(0)

        c_source = self.SE.speakers['hilde'].unsqueeze(0)
        
        # target data
        if target in self.SE.speakers.keys():
            c_target = self.SE.speakers[target].unsqueeze(0)
        else:
            audio_trg = Audio(target, sr)
            audio_trg.preprocess(*preprocess, **preprocess_args)
            c_target = self.SE.embed_utterance(audio_trg.wav).unsqueeze(0)
            

             
        # create spectrogram 
        mel_spec = spectrogram.mel_spec_auto_encoder(audio_src.wav, **kwargs)

        # generate waveform
        if kwargs.get("cut", False):
            mel_batch = torch.stack(mel_spec) # stack list of mel slices

            post_out = self.AE.batch_forward(mel_batch.to(device),
                                             c_source.to(device),
                                             c_target.to(device),
                                             overlap = kwargs.get("overlap", 0.5)) # overlap set to compute_partial_slices default, as they must be equal
            waveform = self.vocoder.generate(post_out.unsqueeze(0))
        else:
            out, post_out, content_codes = self.AE(mel_spec.unsqueeze(0).to(device),
                                                     c_source.to(device),
                                                     c_target.to(device))
            waveform = self.vocoder.generate(post_out)


        # create audio output
        audio_out = Audio(np.asarray(waveform), sr = sr, sr_org = 22050) # where in the vocoder is this sample rate set? 
        audio_out.preprocess(*outprocess, **outprocess_args)

        # return none if save_name is False and a file should not be saved
        if save_name == False:
            return audio_out


        # Generate .wav file frowm waveform
        if save_name is None:
            src_speaker = os.path.splitext(os.path.basename(source))[0]
            trg_speaker = os.path.splitext(os.path.basename(target))[0]
            save_name = f"{src_speaker}_to_{trg_speaker}.wav"
        
        # if out_dir == self.wandb_run and self.wandb_run is not None:
        if save_dir == "wandb":
            # TODO log as table
            assert self.wandb_run is not None, "A wandb run has to be setup with setup_wandb() to save conversion in wandb"
            self.wandb_run.log({save_name.replace(".wav", "") : wandb.Audio(audio_out.wav, caption = save_name, sample_rate = audio_out.sr), **audio_log_dict})
        else:
 
            if save_dir is not None:
                save_dir = save_dir if save_dir.startswith("results") else "results/" + save_dir 
                save_name = save_dir.strip("/") + "/" + save_name
                os.makedirs(save_dir, exist_ok=True) # create folder
            else:
                os.makedirs("results", exist_ok=True) # create folder
                save_name = save_name if save_name.startswith("results") else "results/" + save_name 

            sf.write(save_name, audio_out.wav, samplerate =audio_out.sr)
            print(f"Converted voice has been saved to '{save_name}'")
    
        return audio_out

    def train(self, 
        data_path,
        model_type = VoiceConverterParams["train"]["model_type"], 
        source_examples = VoiceConverterParams["train"]["source_examples"], 
        target_examples = VoiceConverterParams["train"]["target_examples"], 
        **kwargs
    ):
        """
        Trains a model

        Parameters
        ---------
        model_type:
            which model type to train, can be one of ['auto_encoder', 'speaker_encoder']
        source_examples:
            files to use as sources for converting after each epoch
        target_examples:
            files to use as targets for converting after each epoch

        # make list of possible kwargs divided in learn, data loader, data set, if no overlaps do like preprocess and use annotations

        """
        # asssertions, errors and warnings
        if model_type not in ["auto_encoder", "speaker_encoder"]:
            raise ValueError(f"'{model_type}' is not a supported model_type")
        
        convert_examples = source_examples and target_examples
        if not convert_examples:
            print(pformat.YELLOW, "WARN: No conversion examples will be provided as both source_examples and target_examples must be different from None/False", pformat.END)

        # get dataset and learn function of model
        if model_type == "auto_encoder":
            Dataset = AutoEncoderDataset
            learn = self.AE.learn           
        else:
            Dataset = SpeakerEncoderDataset
            learn = self.SE.learn

        # update config
        if model_type == "speaker_encoder" and not isinstance(data_path, dict):
            try:
                self.config[model_type]["dataset"].update({"data_path" : {key.strip():val.strip() for key, val in [arg.split("=") for arg in data_path]}})
            except:
                raise ValueError("data_path must be a dict or list of strings with 'name=' as prefix")
        else:
            self.config[model_type]["dataset"].update({"data_path" : data_path})
        
        for key, value in kwargs.items():
            if key in Dataset.__allowed_args__ + Dataset.__allowed_kw__:
                self.config[model_type]["dataset"].update({key : value})
            elif key in Dataset.get_dataloader.__allowed_args__ + Dataset.get_dataloader.__allowed_kw__:
                self.config[model_type]["dataloader"].update({key : value})
            elif key in learn.__allowed_args__:
                self.config[model_type]["learn"].update({key : value})
            elif key in learn.__allowed_kw__:
                self.config[model_type]["optimizer"].update({key : value})
            else:
                raise ValueError(f"'{key}' is not a valid key word argument")

        # create a wandb run
        self.setup_wandb()
        if model_type == "auto_encoder":
            self.config[model_type]["dataset"].update({"speaker_encoder": self.SE}) # add se to params manually as wandb config converts to string
        

        # train
        if self.verbose:
            start_time = time.time()
            print(f"Starting to train {model_type}...")
        dataset = Dataset(**self.config[model_type]["dataset"])
        dataloader = dataset.get_dataloader(**self.config[model_type]["dataloader"])
        if not convert_examples:
            learn(dataloader, wandb_run = self.wandb_run, **self.config[model_type]["learn"], **self.config[model_type]["optimizer"])
        else:
            learn(
                dataloader, 
                wandb_run = self.wandb_run, 
                voice_converter = self,
                source_examples = source_examples,
                target_examples = target_examples,
                **self.config[model_type]["learn"], 
                **self.config[model_type]["optimizer"]
            )

        if self.verbose:
            print(f"Training finished in {time.time() - start_time}")

        # if model_type == "auto_encoder":
        #     # print("Training device: ", self.AE.params.device)
        #     params = AutoEncoderParams().update(self.config[model_params])
        #     # dataset = TrainDataLoader(**params.get_collection("dataset"), speaker_encoder = self.SE)
        #     # dataset = TrainDataLoader(speaker_encoder = self.SE, chop = True, data_path = 'data/test_data')
        #     dataloader = dataset.get_dataloader(**params.get_collection("dataloader"))
        #     self.AE.learn(dataloader, n_epochs = n_epochs, wandb_run = self.wandb_run)
        # elif model_type == "speaker_encoder":
        #     datadir = {'hilde': ['data/test_data/hilde_7sek'], 'hague': ['data/test_data/HaegueYang_10sek']}
        #     dataset = SpeakerEncoderDataLoader(datadir, device = self.config.get("device", 'cuda'))
        #     dataloader = dataset.get_dataloader(batch_size = 1024)
        #     self.SE.learn(dataloader, n_epochs = 128, wandb_run = self.wandb_run,  log_freq = 16, save_freq = 32)
        #     # raise NotImplementedError()
            
        
        # print(f"Training finished in {time.time() - start_time}")

        # conversion example
        # if convert_examples:
            # self.convert_multiple(
            #     sources = source_examples,
            #     targets = target_examples,
            #     save_dir = "wandb" if not self.wandb_run.mode == "disabled" else "training_examples",
            #     preprocess = [], # no preprocessing of example conversions
            #     out_process = [],
            # )

        # if conversion_examples is None:
        #     self.convert(
        #         # "data/samples/mette_183.wav", 
        #         # "data/samples/chooped7.wav",
        #         "data/samples/hilde_1.wav", 
        #         "data/samples/HaegueYang_5.wav",
        #         save_dir = "wandb" if not self.wandb_run.mode == "disabled" else ".",
        #         # out_folder=os.path.join(self.wandb_run.dir, "conversions")
        #     )
        #     # self.wandb_run.log({"example" : wandb.Audio(wav, caption = "test", sample_rate = sr)})
        # elif conversion_examples:
        #     self.convert_multiple(
        #         conversion_examples[0], 
        #         conversion_examples[1], 
        #         save_dir = "wandb" if not self.wandb_run.mode == "disabled" else ".",
        #     )



    def convert_multiple(self, sources, targets, match_method = "all_combinations", bidirectional = False, **convert_params):
        """
        Uses the convert function on multiple files

        Params
        ------
        sources:
            path or list of paths to source files
        targets:
            path or list of paths to target files
        match_method:
            tells how to match sources and targets
            'all_combinations' will match each source with each target
            'align' will match sources and targets with same index in the given list
        bidirectional:
            boolean telling if targets should also be converted to the source.
        **convert_params
            params to give to VoiceConverter.convert
        """

        # prepare sources
        sources = retrieve_file_paths(sources)
        
        # prepare targets
        targets_args = [targets] if isinstance(targets, str) else targets
        targets = []
        # targets = [retrieve_file_paths(target) if target not in self.SE.speakers.keys else target for target in targets]
        for target in targets_args:
            if target in self.SE.speakers.keys():
                assert not bidirectional, "Cannot convert in both ways if a mean speaker embedding is to be used, as this cannot be the source of a conversion."
                targets.append(target)
            else:
                targets.extend(retrieve_file_paths(target))

        # create empty list for converted audio objects
        audio_objects = []

        # retrieve combinations of sources and targets
        if match_method == "align":
            assert len(sources) == len(targets)
            matches = [*zip(sources, targets)]
        elif match_method == "all_combinations":
            matches = product(sources, targets)

        # make conversions
        for source, target in matches:
            audio_objects.append(self.convert(source, target, **convert_params))
        
        # convert targets to sources if bidirectional is True
        if bidirectional:
            audio_objects.extend(self.convert_multiple(target, sources, match_method, **convert_params))

        return audio_objects

    def learn_speakers(self, mean_speaker_path, mean_speaker_path_excluded = []):
        """
        Learns mean speaker embeddings.
        """
        if isinstance(mean_speaker_path, dict):
            self.config["speaker_encoder"].update({"mean_speakers" : {key:val for key, val in mean_speaker_path.items()}})
        else:
            try:
                self.config["speaker_encoder"].update({"mean_speakers" : {key.strip():val.strip() for key, val in [arg.split("=") for arg in mean_speaker_path]}})
            except:
                raise ValueError("data_path must be a dict or list of strings with 'name=' as prefix")
        
        for speaker, speaker_path in  self.config["speaker_encoder"]["mean_speakers"].items():
            if not speaker_path == "learned_previously":
                self.SE.learn_speaker(speaker, speaker_path, mean_speaker_path_excluded)


    
    def setup_wandb(self, **params):
        """Creates a wandb run. **params are givven to wandb.init()"""
        # skip if setup has already been done
        if self.wandb_run is not None:
            return None

        # overwrite defaults with parsed arguments
        self.config["wandb"].update(params)
        self.config["wandb"].update({"dir" : "logs/" + self.config["wandb"].get("project")}) # create log dir

        # create directory for logs if first run in project
        os.makedirs(self.config["wandb"]["dir"], exist_ok=True)

        # init wandb       
        self.wandb_run = wandb.init(**self.config["wandb"], config = self.config)

        # setup wandb config
        self.config = self.wandb_run.config

    
    def close(self):
        """
        Closes the necesary processes
        """
        if self.wandb_run is not None:
            self.wandb_run.finish()
        wandb.finish()

    def __init_models(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        type_to_artifact = {"auto_encoder" : "AutoEncoder", "speaker_encoder" : "SpeakerEncoder"}
        # model_types, model_names = self.config["model_names"].items()
        model_types, model_names, model_dirs, params  = [], [], [], []
        for model_type, model_name in self.config["model_names"].items():
            # find parameter values
            model_params = self.config.get(f"{model_type}")
            model_dir = model_params["model_dir"]

            


            # check if given model name and dir are valid or look in wandb otherwise
            model_path = model_dir.strip("/") + "/" + model_name
            if not os.path.isfile(model_path):
                if self.config["wandb"]["mode"] in ["disabled", "offline"]:
                    raise ValueError(f"Model {model_path} was not found locally and it was not possible to look in wandb, as the wandb mode was not set to 'online'")
                
                # start looking in wandb
                print(f"Looking for model {model_name} in wandb...")
                self.setup_wandb()
                artifact = self.wandb_run.use_artifact(model_name, type=type_to_artifact[model_type])

                # check if model already exists
                model_name = os.path.basename(artifact.file())
                model_dir = model_dir.strip("/") + "/artifacts"
                model_path = model_dir +"/" + model_name
                # if os.path.isfile(model_path):
                #     raise ValueError(f"The model '{model_path}' already exists, please use this path instead or delete the file if you want to use the model from wandb")

                # download model
                artifact.download(model_dir)

            # append to lists
            model_names.append(model_name)
            params.append(model_params["model"])
            model_dirs.append(model_dir)
            model_types.append(model_type)
        
        self.AE, self.SE, self.vocoder = load_models(
            model_types= model_types,
            model_names = model_names,
            model_dirs = model_dirs,
            params = params,
            device= self.config["device"]
        )
        self.AE = self.AE.to(device)



    def setup_audio_processing_pipeline(self):
        pass


if __name__ == "__main__":

    vc = VoiceConverter(auto_encoder = 'deep_voice_inc/NewSpeaker/Louise2:v0', wandb_params={"project" : "Trials", "name":"Short"})


    # from autovc.utils.argparser import parse_vc_args
    # args = "-mode train -model_type auto_encoder -wandb_params mode=online -n_epochs 1 -data_path data/samples -data_path_excluded data/samples/chooped7.wav -auto_encoder deep_voice_inc/SpeakerEncoder/model_20220113.pt:v4"
    # args = "-mode convert -sources data/samples/hilde_301.wav -targets data/samples/chooped7.wav -convert_params pipes={output:[normalize_volume,remove_noise],source:[normalize_volume]} -auto_encoder models/AutoVC/AutoVC_SMK.pt"# -auto_encoder deep_voice_inc/AutoEncoder/model_20220114.pt:v0"
    # args = "-mode train -model_type auto_encoder -wandb_params mode=online project=SpeakerEncoder2 -data_path data/test_data -n_epochs 16 -speaker_encoder SpeakerEncoder3.pt -auto_encoder_params batch_size=32 chop=True speakers=True save_freq=64 freq=80 "
    # args = "-mode train -model_type speaker_encoder -wandb_params mode=online project=SpeakerEncoder2 -data_path data/test_data -n_epochs 128 -speaker_encoder_params SE_model=models/SpeakerEncoder/SpeakerEncoder.pt"
    # args = "-mode convert -sources data/samples/mette_183.wav -targets data/samples/chooped7.wav"
    # args = None # make sure this is used when not testing
    # args = vars(parse_vc_args(args))

    # vc = VoiceConverter(
    #     auto_encoder=args.pop("auto_encoder", None),
    #     speaker_encoder= args.pop("speaker_encoder", None),
    #     vocoder= args.pop("vocoder", None),
    #     device = args.pop("device", None),
    #     verbose=args.pop("verbose", None),
    #     auto_encoder_params = args.pop("auto_encoder_params", {}),
    #     speaker_encoder_params = args.pop("speaker_encoder_params", {}),
    #     vocoder_params = args.pop("vocoder_params", {}),
    #     wandb_params = args.pop("wandb_params", {})
    # )

    # # get mode
    # mode = args.pop("mode")

    # if mode == "train":
    #     assert all([args.__contains__(key) for key in ["model_type", "n_epochs"]])
    #     convert_examples = all([args.__contains__(key) for key in ["conversions_sources", "conversions_targets"]])
    #     # train
    #     vc.train(
    #         **args
    #         # model_type = args.model_type, 
    #         # n_epochs = args.n_epochs, 
    #         # data_path=args.data_path,
    #         # conversion_examples= None if not convert_examples else [args.conversion_sources, args.conversion_targets]
    #     )
    # elif mode == "convert":
    #     assert all([args.__contains__(key) for key in ["sources", "targets"]])
    #     convert_params = args.pop("convert_params", {})
    #     vc.convert_multiple(
    #         **args, **convert_params
    #         # sources = args.conversion_sources, 
    #         # targets = args.conversion_targets,
    #         # save_folder = args.results_dir,
    #         # bidirectional = args.bidirectional,
    #         # method = args.conversion_data_alignment

    #     )

    
    # vc.close()















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
    #     save_dir = "test"    
    # )

    # vc.train(data = "data/SMK_train/newest_trial", n_epochs = 1)
    # vc.setup_wandb_run(name = "SMK")
    # vc.train(data = "data/SMK_train/20211104", n_epochs = 1)
    # vc.train(data = "data/SMK_train", n_epochs = 1)
    vc.train(data_path = "data/newest_trial", n_epochs=10, model_name="SMK_trial_short_20220125.pt", data_path_excluded="data/newest_trial/hilde")
    # vc.convert("data/samples/mette_183.wav", "data/samples/chooped7.wav")


    vc.close()