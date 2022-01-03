"""
File for storing all hyperparameters and kwargs.
Params are stored as a dictionary, so that they can easily be passed to the functions and classes that needs them. 
These default params shpould not be changed unless a better combination has been found, thus testing of new params should be passed in the functions.
"""

class ClassProperty(object):
    def __init__(self, func):
        self.func = func
    def __get__(self, inst, cls):
        return self.func(cls)



class AutoVCParams:
	# Vocoder
	name 							= "wavenet_vocoder"
	builder 						= "wavenet"
	input_type 						= "raw"
	quantize_channels 				= 65536  # 65536 or 256

	# Audio
	sample_rate 					= 16000
	silence_threshold 				= 2 # this is only valid for mulaw is True
	num_mels 						= 80
	fmin 							= 90
	fmax 							= 7600
	fft_size 						= 1024
	win_length 						= 1024
	window 							= 'hann'
	power							= 1
	hop_size 						=  256
	min_level_db 					=  -100
	ref_level_db 					= 16
	rescaling 						= True # x is an input waveform and rescaled by x / np.abs(x).max() * rescaling_max
	rescaling_max 					=  0.999
	allow_clipping_in_normalization = True # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may happen depends on min_level_db and ref_level_db, causing clipping noise. If False, assertion is added to ensure no clipping happens.o0

	# Model
	log_scale_min 					=  float(-32.23619130191664)
	out_channels 					= 10 * 3
	layers 							= 24
	stacks 							= 4
	residual_channels 				= 512
	gate_channels 					= 512  # split into 2 gropus internally for gated activation
	skip_out_channels 				= 256
	dropout 						= 1 - 0.95
	kernel_size 					= 3
	weight_normalization			= True # If True, apply weight normalization as same as DeepVoice3
	legacy 							= True # Use legacy code or not. Default is True since we already provided a modelbased on the legacy code that can generate high-quality audio. Ref: https://github.com/r9y9/wavenet_vocoder/pull/73

	# Local conditioning (set negative value to disable))
	cin_channels 					= 80 
	upsample_conditional_features 	= True # If True, use transposed convolutions to upsample conditional features, otherwise repeat features to adjust time resolution
	upsample_scales 				= [4, 4, 4, 4] # should np.prod(upsample_scales) == hop_size
	freq_axis_kernel_size 			= 3 # Freq axis kernel size for upsampling network

	# Global conditioning (set negative value to disable)
	# currently limited for speaker embedding this should only be enabled for multi-speaker dataset
	gin_channels 					= -1  # i.e., speaker embedding dim
	n_speakers 						= -1

	# Training:
	batch_size 						= 2,
	adam_beta1 						= 0.9,
	adam_beta2 						= 0.999,
	adam_eps 						= 1e-8,
	amsgrad 						= False
	initial_learning_rate 			= 1e-3
	lr_schedule 					= "noam_learning_rate_decay" # see lrschedule.py for available lr_schedule
	weight_decay 					= 0.0
	clip_thresh 					= -1
	# max time steps can either be specified as sec or steps if both are None, then full audio samples are used in a batch
	max_time_sec 					=  None
	max_time_steps 					= 8000
	exponential_moving_average 		= True # Hold moving averaged parameters and use them for evaluation
	ema_decay 						= 0.9999 # averaged = decay * averaged + (1 - decay) * x

############### WAVE RNN ###############

class WaveRNNParams:
	sample_rate			= 22050
	n_fft 				= 2048
	fft_bins 			= n_fft // 2 + 1
	hop_length			= 275  # 12.5ms - in line with Tacotron 2 paper
	win_length 			= 1100  # 50ms - same reason as above
	fmin 				= 40
	min_level_db 		= -100
	ref_level_db 		= 20

	# Model
	rnn_dims 			= 512
	fc_dims 			= 512
	bits 				= 9
	pad 				= 2  # this will pad the input so that the resnet can 'see' wider than input length
	upsample_factors 	= (5, 5, 11)  # NB - this needs to correctly factorise hop_length
	feat_dims 			= 80 # num_mels
	compute_dims 		= 128
	res_out_dims 		= 128
	res_blocks 			= 10
	mode 				= 'MOL'  # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)

	# Training
	batch_size			= 32
	lr					= 1e-4
	checkpoint_every	= 25_000
	gen_at_checkpoint	= 5  # number of samples to generate at each checkpoint
	total_steps			= 1_000_000  # Total number of training steps
	test_samples		= 50  # How many unseen samples to put aside for testing
	
	seq_len				= hop_length * 5  # must be a multiple of hop_length
	clip_grad_norm 		= 4  # set to None if no gradient clipping needed

	# Generating / Synthesizing
	gen_batched 		= True  # very fast (realtime+) single utterance batched generation
	target 				= 11_000  # target number of samples to be generated in each batch entry
	overlap 			= 550  # number of samples for crossfading between batches
	
	@ClassProperty
	def model(self):
		"""Returns a dictionary with the default model params"""

		keys = ["rnn_dims", "fc_dims", "bits", "pad", "upsample_factors",
                 "feat_dims", "compute_dims", "res_out_dims", "res_blocks",
                 "hop_length", "sample_rate", "mode"
		]
		return {key : self.__getattribute__(self, key) for key in keys}

############### SPEAKER ENCODER ###############

class SpeakerEncoderParams:
	# Mel-filterbank
	mel_window_length 			= 25  # In milliseconds
	mel_window_step 			= 10  # In milliseconds
	mel_n_channels 				= 40

	# Audio
	sampling_rate 				= 16000
	partials_n_frames 			= 160  # x*10 ms.  Number of spectrogram frames in a partial utterance
	inference_n_frames 			= 80  # x*10 ms. Number of spectrogram frames at inference

	# Voice Activation Detection
	vad_window_length 			= 20  # In milliseconds. Window size of the VAD. Must be either 10, 20 or 30 milliseconds. This sets the granularity of the VAD. Should not need to be changed.
	vad_moving_average_width 	= 8# Number of frames to average together when performing the moving average smoothing. The larger this value, the larger the VAD variations must be to not get smoothed out.
	vad_max_silence_length 		= 2 # Maximum number of consecutive silent frames a segment can have.

	# Audio volume normalization
	audio_norm_target_dBFS 		= -30

	# Model parameters
	model_hidden_size 			= 256
	model_embedding_size		= 256
	model_num_layers 			= 3

	# Training parameters
	learning_rate_init 			= 1e-4
	speakers_per_batch 			= 64
	utterances_per_speaker 		= 10

if __name__ == "__main__":
	p = WaveRNNParams.model
	p.update({"sample_rate" : 2})
	print(p)