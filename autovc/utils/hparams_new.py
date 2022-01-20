from datetime import date


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
        "dim_neck"						: 32,
		"dim_emb"						: 256,
		"dim_pre"						: 512,
		"freq" 							: 32,
		"kernel_size" 					: 3,
    },
    "train" : {
        "batch_size" 					: 2,
		"clip_thresh" 					: -1,
		"save_freq"						: 1024,
		"log_freq"						: 8,
		"model_dir"						: "models/AutoVC",
		"model_name"						: "model_" + date.today().strftime("%Y%m%d") +".pt",
		"example_freq"					: None,
        "optimizer" : {
            "betas" : (0.9, 0.999),
            "eps" : 1e-8,
            "amsgrad" : False,
            "lr" : 1e-3,
            "weight_decay" : 0.0
        },
        "ema_decay" : 0.9999
    },

}

SpeakerEncoderParams = {

}

WaveRNNParams = {

}


wandb_params = {
    
}

