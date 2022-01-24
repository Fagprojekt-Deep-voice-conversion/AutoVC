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
    """
    Parse params for Voice Converter init function
    """

    parser = argparse.ArgumentParser(description="Client for parsing arguments to the voice converter", argument_default=argparse.SUPPRESS)  
    

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
    
    # convert multiple params
    parser.add_argument("-sources", nargs='*', type = str, help = "path to source files", required=True)
    parser.add_argument("-targets", nargs='*', type = str, help = "path to target files", required=True)
    # parser.add_argument("-match_method", type = str)
    # parser.add_argument("-bidirectional", type = bool)

    # kwargs
    parser.add_argument("-kwargs", nargs='*', action=ParseKwargs, help = "Other parameters to give to convert")

    # return parser
    if args is None:
        args = parser.parse_args()
        # known_args, unknown_args = parser.parse_known_args()
    else:
        args = parser.parse_args(args.split())
        # known_args, unknown_args = parser.parse_known_args(args.split())

    # return known_args, unknown_args
    return args

def parse_train_args(args = None):
    """
    Parse params for Voice Converter train function
    """

    parser = argparse.ArgumentParser(description="Client for parsing arguments to the voice converter", argument_default=argparse.SUPPRESS)  
    
    # convert multiple params
    parser.add_argument("-data_path", nargs = '*', type = str, help = "path to train data, if model_type is speaker_encoder, paths can have name= in the beginning to indicate the speaker", required=True)
    parser.add_argument("-model_type", choices = ["auto_encoder", "speaker_encoder"], type = str, help = "type of model to train")
    parser.add_argument("-source_examples", nargs='*', type = str, help = "path to source example files")
    parser.add_argument("-target_examples", nargs='*', type = str, help = "path to target example files")

    # kwargs
    parser.add_argument("-kwargs", nargs='*', action=ParseKwargs, help = "Other parameters to give to train")

    # return parser
    if args is None:
        args = parser.parse_args()
        # known_args, unknown_args = parser.parse_known_args()
    else:
        args = parser.parse_args(args.split())
        # known_args, unknown_args = parser.parse_known_args(args.split())

    # return known_args, unknown_args
    return args

if __name__ == "__main__":
    args = "-mode train -data_path hilde=sgasb yang=sfsd hilde=dfsdg"#"# -test False -test2 Ture"
    # print(vars(parse_vc_args(args)))
    args, _ = parse_vc_args(args)

    args = parse_train_args(" ".join(_))
    # print({key : val for arg in vars(args)["data_path"] for key, val in arg.split("=")})
    print({key:val for key, val in [arg.split("=") for arg in vars(args)["data_path"]]})
    # parser = parse_vc_args(args)
    # # print(parser.parse_args(args.split()))
    # print(parser.parse_known_args(args.split()))