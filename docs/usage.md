# Voice Converter usage

## Install 

This package is not supported by pypi and to the repository must therefore be downloaded. After downloading use `cd ...` to navigate to the `AutoVC` folder and type the command

```bash
python -m pip install -e .
```

The `-e` flag can be deprecated as it only tells the package to track changes and is thus only necesary if any modifications are made to any of the scripts.

Furhthermore, requirements can be installed with 

```bash
python -m pip install -r requirements.txt
```

## Python usage

The moduel is first imported and a voice converter object can be created. Here the arguments ending with '_params' are passed as dictionaries:

```python
from autovc import VoiceConverter

# create voice converter object example
vc = VoiceConverter(
    auto_encoder = 'modelname_in_models/AutoVC', 
    # model dir can be set with auto_encoder_params = {"model_dir" : "path/to/model_dir"}
    wandb_params = {"mode":"disabled"} # disable wandb logging
)
```

### Train auto encoder

### Train speaker encoder

### Learn mean speaker embedding

### Single conversion

### Multiple conversions

## Command line tool

## List of parameters

The <b><u>convert</u></b> method allows the parameters listed in [VoiceConverter.convert_multiple](#VoiceConverterconvert_multiple), [VoiceConverter.convert](#VoiceConverterconvert), [mel_spec_auto_encoder](#mel_spec_auto_encoder) and [compute_partial_slices](#compute_partial_slices).

The <b><u>train</u></b> method allows the parameters listed in [VoiceConverter.train](#VoiceConvertertrain),  [compute_partial_slices](#compute_partial_slices), [Dataset.get_dataloader](#Datasetget_dataloader). 
Furthermore, if the model_type is set to 'auto_encoder' the parameters listed in [mel_spec_auto_encoder](#mel_spec_auto_encoder), [AutoEncoder.learn](#AutoEncoderlearn) and [AutoEncoderDataset](#AutoEncoderDataset) can also be passed and if it is 'speaker_encoder' the parameters listed in [mel_spec_speaker_encoder](#mel_spec_speaker_encoder), [SpeakerEncoder.learn](#SpeakerEncoderlearn) and [SpeakerEncoderDataset](#SpeakerEncoderDataset) can also be passed.

The final method <b><u>learn_speakers</u></b> simply takes the parameters 'mean_speaker_path' and 'mean_speaker_path_excluded', which is a dict with speaker as key and path to data for the speaker as value (can also be given as list of strings in the format "speaker=path") and paths to exclude.

### VoiceConverter.convert_multiple
- <b>sources</b>:
    path or list of paths to source files
- <b>targets</b>:
    path or list of paths to target files
- <b>match_method</b>:
    tells how to match sources and targets
    'all_combinations' will match each source with each target
    'align' will match sources and targets with same index in the given list
- <b>bidirectional</b>:
    boolean telling if targets should also be converted to the source.


### VoiceConverter.convert
- <b>sr</b>:
    The sample rate to use for the audio
- <b>save_name</b>:
    Filename for converted sound. 
    If False, no file is saved
- <b>save_dir</b>:
    the folder/directory to store the converted file in, if this is 'wandb' the converted audio will be logged to this run
    all conversions not stored in WANDB will be saved in `results/` folder
- <b>preprocess</b>:
    Pipeline to use for preprocessing (see `autovc.audio.Audio.preprocess`)
- <b>preprocess_args</b>:
    Pipeline args to use for preprocessing (see `autovc.audio.Audio.preprocess`)
- <b>outprocess</b>:
    Pipeline to use for outprocessing (see `autovc.audio.Audio.preprocess`)
- <b>outprocess_args</b>:
    Pipeline args to use for outprocessing (see `autovc.audio.Audio.preprocess`)

### VoiceConverter.train
- <b>model_type</b>:
    which model type to train, can be one of ['auto_encoder', 'speaker_encoder']
- <b>source_examples</b>:
    files to use as sources for converting after each epoch
- <b>target_examples</b>:
    files to use as targets for converting after each epoch


### mel_spec_auto_encoder
- <b>cut</b>:
    If true `compute_partial_slices()` is used to cut the mel spectrogram in slices
- <b>n_fft</b>:
    The lenght of the window - how many samples to include in Fourier Transformation. See librosa for more info.
- <b>hop_length</b>:
    Number of audio samples between adjacent STFT columns - how far the window moves for each FT. See librosa for more info.
- <b>window_length</b>:
    Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft.
- <b>fmin</b>:
    The minimum frequency. See librosa.filters.mel for details.

### mel_spec_speaker_encoder
- <b>mel_window_length</b>:
    In ms. Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft.
- <b>mel_window_step</b>:
    In ms. Used to calculate hop length
- <b>cut</b>:
    If true `compute_partial_slices()` is used to cut the mel spectrogram in slices
- <b>return_slices</b>:
    If true, the computed slices are also returned

### compute_partial_slices
- <b>partial_utterance_n_frames</b>: 
    The number of mel spectrogram frames in each partial utterance (x*10 ms).
    For 1 second samples use partial_utterance_n_frames = 1000/mel_window_step
    Default value chosen from old Speaker Encoder params
- <b>min_pad_coverage</b>: 
    When reaching the last partial utterance, it may or may not have 
    enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present, 
    then the last partial utterance will be considered, as if we padded the audio. 
    Otherwise, it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial 
    utterance, this parameter is ignored so that the function always returns at least 1 slice.
- <b>overlap</b>: 
    By how much the partial utterance should overlap. If set to 0, the partial utterances are entirely disjoint. 


### AutoEncoder.learn
- <b>n_epochs</b>:
    how many epochs to train the model for
- <b>ema_decay</b>:
    ema decay value
- <b>log_freq</b>:
    number of steps between logging the loss to wandb
- <b>save_freq</b>:
    number of epochs between each save of the model
- <b>model_name</b>:
    Name to save the model as, defaults to 'model_xxxxxx.pt'
- <b>save_dir</b>:
    Directory to save the model in, default to 'models/AutoVC
- <b>lr_scheduler</b>:
    name of a learning rate scheduler in `autovc.utils.lr_scheduler`
- <b>n_warmup_steps</b>:
    Number of training steps to take before applying the learning rate schedule

### SpeakerEncoder.learn
- <b>n_epochs</b>:
    how many epochs to train the model for
- <b>log_freq</b>:
    number of steps between logging the loss to wandb
- <b>save_freq</b>:
    number of epochs between each save of the model
- <b>model_name</b>:
    Name to save the model as, defaults to 'model_xxxxxx.pt'
- <b>save_dir</b>:
    Directory to save the model in, default to 'models/AutoVC
- <b>lr_scheduler</b>:
    name of a learning rate scheduler in `autovc.utils.lr_scheduler`
- <b>n_warmup_steps</b>:
    Number of training steps to take before applying the learning rate schedule

Furthermore, `torch.optim.Adam` parameters can also be used.

### AutoEncoderDataset
- <b>data_path</b>:
    Path to the data. See `utils.retrieve_file_paths()` for input format.
- <b>data_path_excluded</b>:
    Paths to exclude from the data. Same as excluded in  `utils.retrieve_file_paths()`.
- <b>use_mean_speaker_embedding</b>:
    If true, the name of the wav file will be compared to the speaker names found in speaker_encoder.speakers.keys(), 
    if the speaker name is found in the file name, the matching mean speaker embedding will be used.
    To use this proparly, the speaker_encoder should learn the necesary mean speaker embedding before creating the data set.

### SpeakerEncoderDataset
- <b>data_path</b>:
    dictionary with speaker name as key and data directory or list of data as key.
- <b>data_path_excluded</b>:
    List of files to exclude from the data loader

### Dataset.get_dataloader
- <b>batch_size</b>:
    Batch size to use
- <b>shuffle</b>:
    bool telling whether or not to shuffle the data

Furthermore, `torch.DataLoader()` params can be used (see [documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)).
