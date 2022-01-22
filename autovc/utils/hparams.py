from datetime import date
from autovc.utils.lr_scheduler import NoamScheduler

AutoEncoderParams = {
	"model_dir"	: "models/AutoVC", # model dir to load from
	"spectrogram" : {
		"sr" : 22050,
		"n_mels" : 80,
		"n_fft" : 2048,
		"hop_length" : 275,  # 12.5ms - in line with Tacotron 2 paper
		"window_length" : 1100,  # 50ms - same reason as above
		"fmin" : 40,
		"mel_window_step" : 12.5,
		"partial_utterance_n_frames" : 400, # corresponding to around 5s
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
		"model_name" : "model_" + date.today().strftime("%Y%m%d") +".pt",
		"save_dir"	: "models/AutoVC", # model dir to save to
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
	"model_dir"	: "models/SpeakerEncoder", # model dir to load from
	"spectrogram" : {
		"sr" : 16000,
		"n_mels" : 40,
		"mel_window_step" : 10,
		"mel_window_length" : 25,
		"partial_utterance_n_frames" : 160, 
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
		"model_name" : "model_" + date.today().strftime("%Y%m%d") +".pt",
		"save_dir"	: "models/SpeakerEncoder", # model dir to save to
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
    # "data_loader" : {
    #         "batch_size" : 2,
    #     },
}

WaveRNNParams = {
	"model_dir"	: "models/WaveRNN",
	"model" : {
		# "sr"	: 22050,
		"hop_length"	: 275,  # 12.5ms - in line with Tacotron 2 paper
		"rnn_dims" 		: 512,
		"res_out_dims" 	: 128,
		"feat_dims" 	: 80, # num_mels
		"fc_dims" 		: 512,
		"bits" 			: 9,
		"upsample_factors" : (5, 5, 11),  # NB - this needs to correctly factorise hop_length. OBS previously called upsample scale in net code
		"compute_dims" 	: 128,
		"pad" 			: 2,  # this will pad the input so that the resnet can 'see' wider than input length
		"res_blocks" 	: 10,
		"mode" 			: 'MOL',  # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
	},
	"generate" : {
		"batched" 		: True,  # very fast (realtime+) single utterance batched generation
		"target" 		: 11_000,  # target number of samples to be generated in each batch entry
		"overlap" 		: 550,  # number of samples for crossfading between batches
		"mu_law" 		: False, # whether to use mu_law
	},

}


WandbParams = {
	# "sync_tensorboard":True, 
	"reinit":True,
	"entity" : "deep_voice_inc",
	# "name" : self.run_name,
	"project" : "GettingStarted", # wandb project name, each project correpsonds to an experiment
	# "dir" : "logs/" + "GetStarted", # dir to store the run in
	# "group" : self.agent_name, # uses the name of the agent class
	"save_code" : True,
	"mode" : "online",
}


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == "__main__":
    
    print(Namespace(**AutoEncoderParams).train.optimizer)

    