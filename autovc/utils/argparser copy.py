import argparse

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

# class ParseListKwargs(argparse.Action):
#     """
#     An argparser action which appends kwargs to a list

#     Example:
#     -PPO policy=MlpPolicy learning_rate=1e-4 
#     For this we will have arg_name = PPO, self.dest=agents, values = [policy=MlpPolicy, learning_rate=1e-4 ]
#     """
#     def __call__(self, parser, namespace, values, option_string=None):
#         arg_name = option_string.replace("-", "") # argument name

#         # set the attribute of the destination to be a list containing a dictionary with the arg_name as key
#         # dest_vals = getattr(namespace, self.dest) if getattr(namespace, self.dest) is not None else []
#         # dest_vals = [] if (getattr(namespace, self.dest)) is None else getattr(namespace, self.dest)
#         # setattr(namespace, self.dest, dest_vals)
        
#         if not hasattr(namespace, self.dest):
#             setattr(namespace, self.dest, [])
#         elif getattr(namespace, self.dest) is None:
#             setattr(namespace, self.dest, [])

#         # get new values to be added to the arg_name dictionary in the destination list
#         new_dest_vals = {}
#         for value in values:
#             key, value = value.split('=')
#             try:
#                 new_dest_vals[key] = eval(value, UndefSubst(locals())) 
#             except:
#                 new_dest_vals[key] = value
        
#         # append new values to the arg_name dictionary
#         getattr(namespace, self.dest).append({arg_name: new_dest_vals})



def parse_vc_args(args = None):
    parser = argparse.ArgumentParser(description="Client for parsing arguments to the voice converter", argument_default=argparse.SUPPRESS)  
    
    parser.add_argument("-mode", choices=["train", "convert"], required=True)
    parser.add_argument("-device", action=StringToNone)#, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)

    # conversion data
    parser.add_argument("-sources", nargs='*', type = str, help = "path to source files")
    parser.add_argument("-targets", nargs='*', type = str, help = "path to target files")
    

    # models
    parser.add_argument("-auto_encoder", action=StringToNone)#, default=None)
    parser.add_argument("-speaker_encoder", action=StringToNone)#, default=None)
    parser.add_argument("-vocoder", action=StringToNone)#, default=None)

    # param dictionaries - train and conversions params for the models are also specified here
    parser.add_argument("-auto_encoder_params", nargs='*', action=ParseKwargs, help = "Auto Encoder parameters")#, default = {})
    parser.add_argument("-speaker_encoder_params", nargs='*', action=ParseKwargs, help = "Speaker Encoder parameters")#, default = {})
    parser.add_argument("-vocoder_params", nargs='*', action=ParseKwargs, help = "Vocoder parameters")#, default = {})
    parser.add_argument("-wandb_params", nargs='*', action=ParseKwargs, help = "WANDB parameters")#, default = {})

    # parser.add_argument("-auto_encoder_params", nargs='*', action=ParseListKwargs, help = "Auto Encoder parameters", dest = "params")
    # parser.add_argument("-speaker_encoder_params", nargs='*', action=ParseListKwargs, help = "Speaker Encoder parameters", dest = "params")
    # parser.add_argument("-vocoder_params", nargs='*', action=ParseListKwargs, help = "Vocoder parameters", dest = "params")
    # parser.add_argument("-wandb_params", nargs='*', action=ParseListKwargs, help = "WANDB parameters", dest = "params")

    # training
    parser.add_argument("-model_type", type = str)
    parser.add_argument("-n_epochs", type = int)
    parser.add_argument("-data_path", nargs='*', type = str)

    # convert
    parser.add_argument("-results_dir", action=StringToNone, help = "directory to store converted audio in")#default = None,)
    parser.add_argument("-bidirectional", type = bool, help = "whether to also convert the targets to the sources") #default = False,
    parser.add_argument("-match_method", type = str, help = "how to match each source to a target (same as 'method' in VoiceConverter.convert)")
    parser.add_argument("-convert_params", nargs='*', action=ParseKwargs)

    # data
    parser.add_argument("-data_path_excluded", nargs='*', type = str)
    
    # return parser

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args.split())

    return args


if __name__ == "__main__":
    args = "-mode train -convert_params pipes={source:[normalize_volume],output:[normalize_volume,remove_noise]}"#"# -test False -test2 Ture"
    print(vars(parse_vc_args(args)))

    # parser = parse_vc_args(args)
    # # print(parser.parse_args(args.split()))
    # print(parser.parse_known_args(args.split()))