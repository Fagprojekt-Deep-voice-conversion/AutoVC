"""
Sets up a command line tool for AutoVC
"""

from autovc.utils.argparser import parse_vc_args, parse_convert_args, parse_train_args
from autovc import VoiceConverter

vc_args, other_args = parse_vc_args()
vc_args = vars(vc_args)

mode = vc_args.pop("mode")
vc = VoiceConverter(**vc_args)


if mode == "train":
    train_args = vars(parse_train_args(" ".join(other_args)))
    # if train_args.get("model_type", "auto_encoder"): # auto encoder is default in vc.train
    #     train_args["data_path"] = {key:val for key, val in [arg.split("=") for arg in train_args["data_path"]]}
    vc.train(**train_args)

elif mode == "convert":
    convert_args = vars(parse_convert_args(" ".join(other_args)))
    vc.convert_multiple(**convert_args)

vc.close()



# if __name__ == "__main__":
#     args = vars(parse_vc_args())

#     vc = VoiceConverter(
#         auto_encoder=args.pop("auto_encoder", None),
#         speaker_encoder= args.pop("speaker_encoder", None),
#         vocoder= args.pop("vocoder", None),
#         device = args.pop("device", None),
#         verbose=args.pop("verbose", None),
#         auto_encoder_params = args.pop("auto_encoder_params", {}),
#         speaker_encoder_params = args.pop("speaker_encoder_params", {}),
#         vocoder_params = args.pop("vocoder_params", {}),
#         wandb_params = args.pop("wandb_params", {})
#     )

#     # get mode
#     mode = args.pop("mode")

#     if mode == "train":
#         assert all([args.__contains__(key) for key in ["model_type", "n_epochs"]])
#         convert_examples = all([args.__contains__(key) for key in ["conversions_sources", "conversions_targets"]])
#         # train
#         vc.train(
#             **args
#         )
#     elif mode == "convert":
#         assert all([args.__contains__(key) for key in ["sources", "targets"]])
#         vc.convert_multiple(
#             **args
#         )


#     vc.close()