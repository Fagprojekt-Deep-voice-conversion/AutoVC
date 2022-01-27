import argparse
from autovc.utils.hparams import VoiceConverterParams
# action classes

class UndefSubst(dict):
    """Class for handling undefined vars when parsing lists"""
    def __missing__(self, key):
        return str(key)

class ParseKwargs(argparse.Action):
    """An argparser action that turns arguments into a dictionary"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')  
            try:
                getattr(namespace, self.dest)[key] = eval(value, UndefSubst(locals())) 
            except:
                getattr(namespace, self.dest)[key] = value

class StringToNone(argparse.Action):
    """An argparser action that parses the string "None" as None """
    def __call__(self, parser, namespace, value, option_string=None):
        # setattr(namespace, self.dest, eval(values, UndefSubst(locals())))
        if value == "None":
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, value)


# functions for adding params

def add_init_args(parser):
     # add arguments
    parser.add_argument("-mode", choices=["train", "convert"], required=True)
    parser.add_argument("-device", action=StringToNone)
    parser.add_argument("--verbose", action="store_true", default=False)

    # models
    parser.add_argument("-auto_encoder", action=StringToNone)#, default=None)
    parser.add_argument("-speaker_encoder", action=StringToNone)#, default=None)
    parser.add_argument("-vocoder", action=StringToNone)#, default=None)

    # param dictionaries - train and conversions params for the models are also specified here
    parser.add_argument("-auto_encoder_params", nargs='*', action=ParseKwargs, help = "Auto Encoder parameters")#, default = {})
    parser.add_argument("-speaker_encoder_params", nargs='*', action=ParseKwargs, help = "Speaker Encoder parameters")#, default = {})
    parser.add_argument("-vocoder_params", nargs='*', action=ParseKwargs, help = "Vocoder parameters")#, default = {})
    parser.add_argument("-wandb_params", nargs='*', action=ParseKwargs, help = "WANDB parameters")#, default = {})

    # mean speaker embeddings
    parser.add_argument("-mean_speaker_path", nargs='*', type = str, help = "path to speaker data, should be in the foramt 'speaker=paths', e.g. -mean_speaker_path JohnDoe=path/to/John")
    parser.add_argument("-mean_speaker_path_excluded", nargs='*', type = str, help = "Paths to avoid using when learning the mean speaker embedding")

    return parser

def add_convert_multiple_args(parser):
    parser.add_argument("-sources", nargs='*', type = str, help = "path to source files", required=True)
    parser.add_argument("-targets", nargs='*', type = str, help = "path to target files", required=True)
    parser.add_argument("-match_method", type = str)
    parser.add_argument("-bidirectional", type = bool)

    return parser

def add_convert_args(parser):
    parser.add_argument("-save_name", type = str)
    parser.add_argument('-save_dir', type = str)
    parser.add_argument('-preprocess', nargs = '*', type = str)
    parser.add_argument('-preprocess_args', nargs='*', action=ParseKwargs)
    parser.add_argument('-outprocess', nargs = '*', type = str)
    parser.add_argument('-outprocess_args', nargs='*', action=ParseKwargs)

    return parser

def add_train_args(parser):
    parser.add_argument('-model_type', type = str)
    parser.add_argument('-source_examples', nargs = '*', action=StringToNone)
    parser.add_argument('-target_examples', nargs = '*', action=StringToNone)

    return parser

def add_mel_spec_auto_encoder_args(parser):
    parser.add_argument('-cut', type = bool)
    parser.add_argument('-n_fft', type = int)
    parser.add_argument('-hop_length', type = int)
    parser.add_argument('-window_length', type = int)
    parser.add_argument('-fmin', type = int)

    return parser

def add_mel_spec_speaker_encoder_args(parser):
    parser.add_argument('-mel_window_length', type = int)
    parser.add_argument('-mel_window_step')
    parser.add_argument('-cut', type = bool)
    parser.add_argument('-return_slices', type = bool)

    return parser

def add_compute_partial_slices_args(parser):
    parser.add_argument('-partial_utterance_n_frames', type = int)
    parser.add_argument('-min_pad_coverage', type = float)
    parser.add_argument('-overlap', type = float)

    return parser

def add_learn_args(parser):
    parser.add_argument('-n_epochs', type = int)
    parser.add_argument('-log_freq', type = int)
    parser.add_argument('-save_freq', type = int)
    parser.add_argument('-model_name', type = str)
    parser.add_argument('-save_dir', type = str)
    parser.add_argument('-lr_scheduler', type = str)
    parser.add_argument('-n_warmup_steps', type = int)
    parser.add_argument("-opt_kwargs", nargs='*', action=ParseKwargs, help = "Other parameters to give to Adam optimizer")
    parser.add_argument('-ema_decay', type = float) # only for AE


    return parser

def add_dataset_args(parser):
    parser.add_argument('-data_path', nargs = '*', type = str)
    parser.add_argument('-data_path_excluded', nargs = '*', type = str)
    parser.add_argument('-preprocess', nargs = '*', type = str)
    parser.add_argument('-preprocess_args', nargs='*', action=ParseKwargs)
    parser.add_argument('-use_mean_speaker_embedding', type = bool) # only for AE

    return parser

def add_dataloader_args(parser):
    parser.add_argument('-batch_size', type = int)
    parser.add_argument('-shuffle', type = bool)
    parser.add_argument("-dataloader_kwargs", nargs='*', action=ParseKwargs, help = "Other parameters to give to torch data loader")

    return parser

# parse functions

def parse_vc_args(args = None):
    """
    Parse params for Voice Converter init function
    """

    parser = argparse.ArgumentParser(description="Client for parsing arguments to the voice converter", argument_default=argparse.SUPPRESS)  
    

    # add arguments
    parser = add_init_args(parser)

    # return parser
    if args is None:
        # args = parser.parse_args()
        known_args, unknown_args = parser.parse_known_args()
    else:
        # args = parser.parse_args(args.split())
        known_args, unknown_args = parser.parse_known_args(args.split())

    return known_args, unknown_args



def parse_convert_args(args = None):
    """
    Parse params for Voice Converter convert (multiple) function
    """

    parser = argparse.ArgumentParser(description="Client for parsing arguments to the voice converter", argument_default=argparse.SUPPRESS)  
    
    
    # add arguments
    parser = add_convert_args(parser)
    parser = add_convert_multiple_args(parser)
    parser = add_mel_spec_auto_encoder_args(parser)
    parser = add_compute_partial_slices_args(parser)

    # return parser
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args.split())
    return args

def parse_train_args(args = None):
    """
    Parse params for Voice Converter train function
    """
    # get model type
    type_parser = argparse.ArgumentParser(description="Client for parsing arguments to the voice converter", argument_default=argparse.SUPPRESS)  
    type_parser.add_argument('-model_type', type = str)
    if args is None:
        known_args, unknown_args = type_parser.parse_known_args()
    else:
        known_args, unknown_args = type_parser.parse_known_args(args.split())
    
    # create parser
    parser = argparse.ArgumentParser(description="Client for parsing arguments to the voice converter", argument_default=argparse.SUPPRESS)  

    # add general training args
    parser = add_train_args(parser)
    parser = add_compute_partial_slices_args(parser)
    parser = add_dataloader_args(parser)
    parser = add_dataset_args(parser)
    parser = add_learn_args(parser)

    # add model specific args
    if vars(known_args).get("model_type", VoiceConverterParams["train"]["model_type"]) == "auto_encoder":
        parser = add_mel_spec_auto_encoder_args(parser)
    else:
        parser = add_mel_spec_speaker_encoder_args(parser)
    

    # parse general args
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args.split())

    return args

if __name__ == "__main__":
    args = "-mode train -data_path hilde=sgasb yang=sfsd hilde=dfsdg"#"# -test False -test2 Ture"

    args = " ".join(param.strip() for param in [
    # "-speaker_encoder SpeakerEncoder_SMK.pt",
    "-mode train",
    "-model_type auto_encoder",
    # "-auto_encoder AutoVC_SMK.pt",
    
    # "-auto_encoder_params cut=True speakers=False model_name=SMK_trial_20220125.pt",
    # "-data_path data/newest_trial",
    "-data_path data/hilde_subset.wav data/SMK_HY_long.wav data/yang_long.wav ",
    "-n_epochs 10",
    "-model_name SMK_trial_20220125.pt",
    "-cut True",

    # wandb
    f"-wandb_params project={'project'} name={'test'}"
])

    # print(vars(parse_vc_args(args)))
    args, _ = parse_vc_args(args)

    args = parse_train_args(" ".join(_))
    # print({key : val for arg in vars(args)["data_path"] for key, val in arg.split("=")})
    print(args)
    # parser = parse_vc_args(args)
    # # print(parser.parse_args(args.split()))
    # print(parser.parse_known_args(args.split()))