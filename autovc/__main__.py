"""
Sets up a command line tool for AutoVC
"""

from autovc.utils.argparser import parse_vc_args
from autovc import VoiceConverter

if __name__ == "__main__":
    args = vars(parse_vc_args())

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
        )
    elif mode == "convert":
        assert all([args.__contains__(key) for key in ["sources", "targets"]])
        vc.convert_multiple(
            **args
        )


    vc.close()