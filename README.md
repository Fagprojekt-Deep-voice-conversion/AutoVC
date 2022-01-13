# Auto VC - Voice Conversion with only reconstruction loss
Auto VC is an Autoencoder which trains only on reconstruction loss. This means the Mel-Spectrogram of a speech sound is processed through a 'bottleneck' which downsamples and compressed the input into a latent feature space. The latent features are afterwards upsampled and expandended back to the original space.
The reconstruction loss is simply the similarity between input and output.

The main idea is to loss every information about the speaker identity and then provide this information after the bottleneck. Providing the identity of a different speaker should thus result in voice conversion.


## Install 

This package is not supported by pypi and to the repository must therefore be downloaded. After downloading use `cd ...` to navigate to the `AutoVC` folder and type the command

```bash
pip install -e .
```

The `-e` flag can be deprecated as it only tells the package to track changes and is thus only necesary if any modifications are made to any of the scripts.

## Usage

This package uses [Weights & Biases (wandb)](https://docs.wandb.ai/) for profiling the code and a wandb account is therefore encouraged. If you do not want to create a wandb account, the code can be with `-wandb_params mode=disabled`

Below is a minimal example of how to use both the command line tool and the python class, showing examples of how to train and convert. For a more detailed explanation of how each part of the model works see [docs/usage](docs/usage_old.md)

#### Command line usage:
To train the default auto encoder for 10 epochs on the sample data:
```bash 
python autovc -mode train -model_type auto_encoder -n_epochs 10
```

#### Python usage
To convert two audio files with the python class:
```python
from autovc import VoiceConverter

vc = VoiceConverter()
vc.convert(source, target)
vc.close()
```

## Ethical usage

This package should not be used for harmful or illegal purposes. To avoid this, please make sure that the voices used for conversations can be used under a legal copyright.

## Examples

Examples can be seen in wandb at [link_to_example](wandb/example.com).

## Documentation

For information about the structure of the repo, how to cite and how to use have a look at the `docs/` folder.

