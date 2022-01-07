## How To Use ... 
--------
### Training
To train the model you will need a folder of wav files to train on. As we are doing no real testing, all files could be stored together in one folder, say, ./data. <br/>
You need to write a function which loads this folder into a list of path strings and labels (could just be integers) <br/>
The previously used DataLoad2 is a mess, as we wanted to load in speech of differnt speakers and split this into test sets and only use a specific amount of minuts for each speaker to train on... As this is not needed anymore, the function could simply be rewritten into something like:
```
def dataload(path):
    train_data = os.list(path)
    train_labels = np.arange(len(train_data))
    return train_data, train_labels
```
The function `Train()` found in `Train_and_Loss.py` does this for you. It trains a model, which is a `Generator`object found in `Generator_autoVC/model_vc.py`. The parameters for the Generator have to be specified as `Generator(32, 256, 512, 32).to(device)`
- 32 = Dimension of bottleneck
- 256 = Dimension of speaker embedding
- 512 = Dimension of first LSTM layer in decoder
- 32 = Sampling frequency for encoder downsampling
`Train()` also needs a torch Trainloader object, which samples training batches. This is created by the function `Trainloader` found in `Train_and_Loss.py`.
This function needs:
- `Data`: a list of training files (files are path strings)
- `labels`: a list of training labels (labels can be integers for instance)
- `batch_size` - we used 2 in our project.
- `shuffle` : bool - to shuffle data or not deafult = True  
- `num_workers`: for parallel stuff .... default 1 ....
- `pin_memory`: dunno ... default - False
- `vocoder`: string of which vocoder to use. Deafult "autovc". Just means it uses a vocoder compatible with autoVC.

Before training it is a good idea to load a pretrained model for transfer learning:
```
g_checkpoint = torch.load(pretrained_model_path, map_location=torch.device(device))
model.load_state_dict(g_checkpoint['model_state'])
```
If multiple workers are used also remember to call `model.share_memory()` (Dont think we'll need this).

Only things left to do is to specify other Train params:
- `init_lr`: initial learning rate. Default 1e-3. The initial learning rate for optimnizer. This is decayed using Noam Chomsky.
- `n_steps`: How many training steps to take. If we are training in epochs: n_steps = # files / batch_size * epochs
- `save_every`: save every some step. Saves a model every some step ....
- `models_dir`: The directory where models are stored.
- `loss_dir`: The directory where the training loss is stored. Can be read using pickle and visualised.
- `model_path_name`: The name of the model file
- `loss_path_name`: The name of the loss file

The Train function saves 2 models every some step `save_every` and at the end of training. <br/>
The first is the model at this step. The second is an "average" model, which takes a 'ema'-deacaying average of all previous models parameters ... Maybe more stable? Maybe not? <br/>
NB in the bottom of `Train()` the models are save with some filename. If you wish to change these names please go ahead!

### Converting
It is pretty straight forward to convert speech. <br/>
First use `Instantiate_Models` (found in `conversion.py`) to setup AutoVC (`model`) and a vocoder (`voc_model`). You only need the model path of AutoVC as the vocoder is WaveRNN per default. <br/>
Take to .wav files, a source and a target and call `Zero_shot(source, target, model, voc_model, save_path)`. The conversion is saved as `save_path/conversion.wav`. <br/>
The function `Generate` in `conversion.py` generates the sound after converting with AutoVC. You might want to check this out to alter the name to something else than "conversion".

## REPO overview:
-------
### Generator_autoVC (folder)
- AutoVC_preprocessing.py:
    - Contains functions for preprocessing data into an appropriate format for the AutoVC Model
- model_vc.py:
    - The model structure of AutoVC


### Models (folder)
- Simply contains .pt files of models trained and used in the learning. For instance pretrained vocoder model and speaker identity encoder are found here.

### run_scripts (folder)
- Files used for running a training on HPC

### Speaker_encoder (folder)
- Model structure, preprocessing tools, inference etc. used with the speaker encoder.
- This model is pretrained and should only provide speaker identities to the AutoVC.

### vocoder (folder)
- Model structure, preprocessing tools, inference etc. used with the vocoder.
- This model is pretrained and is only used for generating sound from a mel spectrogram.


### conversion.py
- Instantiate_Models(path): Setups the AutoVC model using a pretrained model from path
- embed_utterance(path): Embeds the speaker identity from a .wav file
- generate(m, path): generates a .wav file to path from mel spectrogram m
- Conversion(source, target, model, vocoder): Converts source speech to target using model and generating speech with vocoder.
- Experiment: for our experiment back then .... I think it is outdated as we wish to overfit the model.

### dataload.py
- Needs a proper clean up! Just needed for loading a list of paths to .wav files .... and give each a label....

### hparams.py
- The hyperparameters of the model - Nix Pille!

### Preprocessing_WAV.py
- Used for creating Mel spectrograms from wav files

### Run_AutoVC.py
- The script used for training.

### SpeakerIdentiy.py
- Computes the speaker identity

### Train_and_loss.py
- trainloader: Creates a torch data loader for training.
    - collate and concat_dataset are helper functions

- loss: The full reconstruction loss for the AutoVC model
- noam_learning_rate_decay: Learning rate schedule for annealing learning rate
- Train: trains the model for n steps with batch size ....

### zero_shot.py
- Provides an easy to use tool for voice conversion.