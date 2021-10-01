# Auto VC - Voice Conversion with only reconstruction loss
Auto VC is an Autoencoder which trains only on reconstruction loss. This means the Mel-Spectrogram of a speech sound is processed through a 'bottleneck' which downsamples and compressed the input into a latent feature space. The latent features are afterwards upsampled and expandended back to the original space.
The reconstruction loss is simply the similarity between input and output.

The main idea is to loss every information about the speaker identity and then provide this information after the bottleneck. Providing the identity of a different speaker should thus result in voice conversion.


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
