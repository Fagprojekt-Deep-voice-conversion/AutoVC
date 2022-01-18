"""
File for storing all hyperparameters and kwargs.
Params are stored as a dictionary, so that they can easily be passed to the functions and classes that needs them. 
These default params shpould not be changed unless a better combination has been found, thus testing of new params should be passed in the functions.
"""

from autovc.utils.lr_scheduler import NoamScheduler
import torch
from datetime import date


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class ClassProperty(object):
    def __init__(self, func):
        self.func = func
    def __get__(self, inst, cls):
        return self.func(cls)

# class ParamCollection(Namespace):
# 	def update(self, d: dict):
# 		self.__dict__.update(d)

class ParamCollection:
	def __init__(self) -> None:
		# raise NotImplementedError
		self.collections = {"all": None}
		self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def update(self, params: dict):		
		self.__dict__.update(params)
		return self

	def get_collection(self, collection = "all"):
		"""
		Returns a collection of parameters as a dictionary
		"""
		# if collection == "all":
		# 	return self.__dict__
		keys = self.collections.get(collection, ValueError("Collection does not exist"))

		if keys == None:
			return self.__dict__
		else:
			return {key : val for key, val in self.__dict__.items() if key in keys}

	def add_collection(self, name, params):
		self.collections[name] = params
	
	def __repr__(self) -> str:
		return self.__dict__.__repr__()

# class AudioParams(ParamCollection):
# 	"""
# 	Class for audio params shared for all models
# 	"""
# 	def __init__(self) -> None:
# 		super().__init__()

class AutoEncoderParams(ParamCollection):
	# Vocoder
	# name 							= "wavenet_vocoder"
	# builder 						= "wavenet"
	# input_type 						= "raw"
	# quantize_channels 				= 65536  # 65536 or 256
	def __init__(self) -> None:
		super().__init__()
		# Audio
		self.sample_rate 					= 16000
		self.silence_threshold 				= 2 # this is only valid for mulaw is True
		self.num_mels 						= 80
		self.fmin 							= 90
		self.fmax 							= 7600
		self.fft_size 						= 1024
		self.win_length 					= 1024
		self.window 						= 'hann'
		self.power							= 1
		self.hop_size 						= 256
		self.min_level_db 					= -100
		self.ref_level_db 					= 16
		self.rescaling 						= True # x is an input waveform and rescaled by x / np.abs(x).max() * rescaling_max
		self.rescaling_max 					=  0.999
		self.allow_clipping_in_normalization = True # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may happen depends on min_level_db and ref_level_db, causing clipping noise. If False, assertion is added to ensure no clipping happens.o0

		# Model
		self.dim_neck						= 32
		self.dim_emb						= 256
		self.dim_pre						= 512
		self.freq 							= 32
		self.log_scale_min 					= float(-32.23619130191664)
		self.out_channels 					= 10 * 3
		self.layers 						= 24
		self.stacks 						= 4
		self.residual_channels 				= 512
		self.gate_channels 					= 512  # split into 2 gropus internally for gated activation
		self.skip_out_channels 				= 256
		self.dropout 						= 1 - 0.95
		self.kernel_size 					= 3
		self.weight_normalization			= True # If True, apply weight normalization as same as DeepVoice3
		self.legacy 						= True # Use legacy code or not. Default is True since we already provided a modelbased on the legacy code that can generate high-quality audio. Ref: https://github.com/r9y9/wavenet_vocoder/pull/73

		# Local conditioning (set negative value to disable))
		self.cin_channels 					= 80 
		self.upsample_conditional_features 	= True # If True, use transposed convolutions to upsample conditional features, otherwise repeat features to adjust time resolution
		self.upsample_scales 				= [4, 4, 4, 4] # should np.prod(upsample_scales) == hop_size
		self.freq_axis_kernel_size 			= 3 # Freq axis kernel size for upsampling network

		# Global conditioning (set negative value to disable)
		# currently limited for speaker embedding this should only be enabled for multi-speaker dataset
		self.gin_channels 					= -1  # i.e., speaker embedding dim
		self.n_speakers 					= -1

		# Training:
		self.batch_size 					= 2
		self.clip_thresh 					= -1
		self.save_freq						= 1024
		self.log_freq						= 1
		self.model_dir						= "models/AutoVC"
		self.model_name						= "model_" + date.today().strftime("%Y%m%d") + ".pt"
		self.example_freq					= None

		# Optimizer
		self.betas 							= (0.9, 0.999)
		self.eps 							= 1e-8
		self.amsgrad 						= False
		self.lr 							= 1e-3 # learning rate
		self.weight_decay 					= 0.0
		
		# Learning rate scheduler
		self.lr_scheduler 					= NoamScheduler # see lrschedule.py for available lr_schedule
		self.dim_model						= 80 # The output dimension of the model
		self.n_warmup_steps					= 200

		# max time steps can either be specified as sec or steps if both are None, then full audio samples are used in a batch
		self.max_time_sec 					=  None
		self.max_time_steps 				= 8000
		self.exponential_moving_average 	= True # Hold moving averaged parameters and use them for evaluation
		self.ema_decay 						= 0.9999 # averaged = decay * averaged + (1 - decay) * x

		# add collections
		self.add_collection("Encoder", ["dim_neck", "dim_emb", "freq"])
		self.add_collection("Decoder", ["dim_neck", "dim_emb", "dim_pre"])
		self.add_collection("Adam", ["betas", "eps", "amsgrad", "lr", "weight_decay"])
		self.add_collection("lr_scheduler", ["dim_model", "n_warmup_steps"])
		self.add_collection("dataset", ["data_path", "data_path_excluded", "chop"])
		self.add_collection("dataloader", ["batch_size", "shuffle", "num_workers", ])
	



############### WAVE RNN ###############

class WaveRNNParams(ParamCollection):
	def __init__(self):
		super().__init__()
		self.sample_rate		= 22050
		self.n_fft 				= 2048
		self.fft_bins 			= self.fft_bins_prop 
		self.hop_length			= 275  # 12.5ms - in line with Tacotron 2 paper
		self.win_length 		= 1100  # 50ms - same reason as above
		self.fmin 				= 40
		self.min_level_db 		= -100
		self.ref_level_db 		= 20
		self.mel_window_step 	= 12.5
		# Model
		self.rnn_dims 			= 512
		self.fc_dims 			= 512
		self.bits 				= 9
		self.pad 				= 2  # this will pad the input so that the resnet can 'see' wider than input length
		self.upsample_factors 	= (5, 5, 11)  # NB - this needs to correctly factorise hop_length. OBS previously called upsample scale in net code
		self.feat_dims 			= 80 # num_mels
		self.compute_dims 		= 128
		self.res_out_dims 		= 128
		self.res_blocks 		= 10
		self.mode 				= 'MOL'  # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)

		# Training
		self.batch_size			= 32
		self.lr					= 1e-4
		self.checkpoint_every	= 25_000
		self.gen_at_checkpoint	= 5  # number of samples to generate at each checkpoint
		self.total_steps		= 1_000_000  # Total number of training steps
		self.test_samples		= 50  # How many unseen samples to put aside for testing
		
		self.seq_len			= self.seq_len_prop  # must be a multiple of hop_length
		self.clip_grad_norm 	= 4  # set to None if no gradient clipping needed

		# Generating / Synthesizing
		self.batched 			= True  # very fast (realtime+) single utterance batched generation
		self.target 			= 11_000  # target number of samples to be generated in each batch entry
		self.overlap 			= 550  # number of samples for crossfading between batches
		self.mu_law 			= False # whether to use mu_law

		# add collections
		self.add_collection("synthesize", ["batched", "target", "overlap", "mu_law"])
		self.add_collection("UpsampleNetwork", ["feat_dims", "upsample_factors", "compute_dims", "res_blocks", "res_out_dims", "pad"])
		# self.add_collection("model", [rnn_dims, fc_dims, bits, pad, upsample_factors, feat_dims, compute_dims, res_out_dims, res_blocks])

	@property
	def fft_bins_prop(self):
		return self.n_fft // 2 + 1

	@property
	def seq_len_prop(self):
		return self.hop_length*5

	def update(self, params: dict):
		super().update(params)
		
		if "n_fft" in params and "fft_bins" not in params:
			self.__dict__["fft_bins"] = self.fft_bins_prop
		
		if "hop_length" in params and "seq_len" not in params:
			self.__dict__["seq_len"] = self.seq_len_prop

		return self
	
	


############### SPEAKER ENCODER ###############

class SpeakerEncoderParams(ParamCollection):
	def __init__(self) -> None:
		super().__init__()

		# Mel-filterbank
		self.mel_window_length 			= 25  # In milliseconds
		self.mel_window_step 			= 10  # In milliseconds
		self.mel_n_channels 			= 40

		# Audio
		self.sampling_rate 				= 16000
		self.partials_n_frames 			= 160  # x*10 ms.  Number of spectrogram frames in a partial utterance
		self.inference_n_frames 		= 80  # x*10 ms. Number of spectrogram frames at inference

		# Voice Activation Detection
		self.vad_window_length 			= 20  # In milliseconds. Window size of the VAD. Must be either 10, 20 or 30 milliseconds. This sets the granularity of the VAD. Should not need to be changed.
		self.vad_moving_average_width 	= 8# Number of frames to average together when performing the moving average smoothing. The larger this value, the larger the VAD variations must be to not get smoothed out.
		self.vad_max_silence_length 	= 2 # Maximum number of consecutive silent frames a segment can have.

		# Audio volume normalization
		self.audio_norm_target_dBFS 	= -30

		# Model parameters
		self.model_hidden_size 			= 256
		self.model_embedding_size		= 256
		self.model_num_layers 			= 3
		self.batch_first				= True

		# Training parameters
		self.learning_rate_init 		= 1e-4
		self.speakers_per_batch 		= 64
		self.utterances_per_speaker 	= 10
		self.log_freq 					= 1
		self.model_dir					= "models/SpeakerEncoder"
		self.model_name					= "model_" + date.today().strftime("%Y%m%d") + ".pt"

		# Learning rate scheduler
		self.lr_scheduler 					= NoamScheduler # see lrschedule.py for available lr_schedule
		self.dim_model						= 256 # The output dimension of the model
		self.n_warmup_steps					= 64

		# Optimizer
		self.betas 							= (0.9, 0.999)
		self.eps 							= 1e-8
		self.amsgrad 						= False
		self.lr 							= 1e-3 # learning rate
		self.weight_decay 					= 0.0

		
		self.add_collection("lr_scheduler", ["dim_model", "n_warmup_steps"])
		self.add_collection("Adam", ["betas", "eps", "amsgrad", "lr", "weight_decay"])

		# add collections
		# self.add_collection("LSTM", [])


# class WandbParams(ParamCollection):
# 	# wand_defaults = {
#     #         # "sync_tensorboard":True, 
#     #         "reinit":True,
#     #         "entity" : "deep_voice_inc",
#     #         # "name" : self.run_name,
#     #         "project" : "GetStarted", # wandb project name, each project correpsonds to an experiment
#     #         "dir" : "logs/" + "GetStarted", # dir to store the run in
#     #         # "group" : self.agent_name, # uses the name of the agent class
#     #         "save_code":True,
#     #         "mode" : "online",
#     #         "config" : self.config,
#     #     }

# 	def __init__(self) -> None:
# 		super().__init__()

# 		self.reinit = True
# 		self.entity = 'deep_voice_inc'
# 		project = "test"


############## VOICE CONVERTER ###############
class VoiceConverterParams(ParamCollection):
	def __init__(self) -> None:
		super().__init__()

		self.AE_model = 'models/AutoVC/AutoVC_seed40_200k.pt'
		self.SE_model = 'models/SpeakerEncoder/SpeakerEncoder.pt'
		self.vocoder_model = 'models/WaveRNN/WaveRNN_Pretrained.pyt'

		self.add_collection("model_names", ["AE_model", "SE_model", "vocoder_model"])

	def update(self, params: dict = {}, auto_encoder_model = None, speaker_encoder_model = None, vocoder_model = None):
		super().update({
			**params, 
			"AE_model" : self.AE_model if auto_encoder_model is None else auto_encoder_model,
			"SE_model" : self.SE_model if speaker_encoder_model is None else speaker_encoder_model,
			"vocoder_model" : self.vocoder_model if vocoder_model is None else vocoder_model,
		})
		return self 

# class VoiceConverterParams:#(ParamCollection):
# 	def __init__(self) -> None:
# 		super().__init__()
# 		self.AE = AutoEncoderParams()
# 		self.SE = SpeakerEncoderParams()
# 		self.vocoder = WaveRNNParams()

# 		# self.default_models = {
# 		# 	"auto_encoder": 'models/AutoVC/autovc_SMK.pt',
# 		# 	"speaker_encoder": ,
# 		# 	"vocoder":
# 		# }
	
# 	def update(self, AE_params = {}, SE_params = {}, vocoder_params = {}, **kwargs):
# 		self.AE.update(AE_params)
# 		self.SE.update(SE_params)
# 		self.vocoder.update(vocoder_params)
# 		self.__dict__.update(kwargs)

# 		return self



if __name__ == "__main__":
	# p = WaveRNNParams.synthesize
	# p.update({"sample_rate" : 2})
	# print(p.__dict__)

	# def update_params(params, new_params):
	# 	for key, val in new_params:
	# 		params().__set

	# p = WaveRNNParams()
	p = VoiceConverterParams()
	# p.__setattr__("sample_rate", 2)
	# p.update({"sample_rate" : 2})
	# p.sample_rate = 2
	# p.__setattr__("sample_rate", 2)

	p.update({"sample_rate" : 2, "n_fft" : 30, "batched" : False})
	# print(p.__dict__)
	# print(p.get_collection("synthesize"))
	# print(p.sample_rate)
	# print(p.model)
	# print(p.test)

	print(p)
