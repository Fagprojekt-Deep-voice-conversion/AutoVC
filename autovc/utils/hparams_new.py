from datetime import date
from autovc.utils.lr_scheduler import NoamScheduler

AutoEncoderParams = {
    "audio" : {
        "sr" : 16000,
        "num_mels" : 80,
        "fmin" : 90,
        "fft_size" : 1024,
        "win_length" : 1024,
        "hop_size" : 256,
        "min_level_db" : -100,
    },
    "model" : {
        "dim_neck" : 32,
		"dim_emb" : 256,
		"dim_pre" : 512,
		"freq" : 32,
    },
    "learn" : {
        "log_freq" : 8,
        "save_freq" : 1024,
		"model_dir"	: "models/AutoVC",
		"model_name" : "model_" + date.today().strftime("%Y%m%d") +".pt",
		"example_freq" : None,
        "ema_decay" : 0.9999,
        "optimizer" : {
            "betas" : (0.9, 0.999),
            "eps" : 1e-8,
            "amsgrad" : False,
            "lr" : 1e-3,
            "weight_decay" : 0.0,
            "lr_scheduler" : NoamScheduler,
		    "n_warmup_steps" : 64,
        },
        

    },
    "data_loader" : {
            "batch_size" : 2,
        },

}

SpeakerEncoderParams = {
    "audio" : {
        "sr" : 16000,
        "num_mels" : 40,
    },
    "model" : {
        "input_size" : 40, # same as number of mels
        "hidden_size" : 256,
		"embedding_size" : 256,
		"num_layers" : 3,
		"batch_first" : True,
    },
    "learn" : {
        "log_freq" : 1,
        "save_freq" : 1024,
		"model_dir"	: "models/SpeakerEncoder",
		"model_name" : "model_" + date.today().strftime("%Y%m%d") +".pt",
		"example_freq" : None,
        "ema_decay" : 0.9999,
        "optimizer" : {
            "betas" : (0.9, 0.999),
            "eps" : 1e-8,
            "amsgrad" : False,
            "lr" : 1e-3,
            "weight_decay" : 0.0,
            "lr_scheduler" : NoamScheduler,
		    "n_warmup_steps" : 64,
        },
        

    },
    "data_loader" : {
            "batch_size" : 2,
        },
}

WaveRNNParams = {

}


wandb_params = {
    
}


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == "__main__":
    
    print(Namespace(**AutoEncoderParams).train.optimizer)

    