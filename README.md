# Auto VC - Voice Conversion with only reconstruction loss
Auto VC is an Autoencoder which trains only on reconstruction loss. This means the Mel-Spectrogram of a speech sound is processed through a 'bottleneck' which downsamples and compressed the input into a latent feature space. The latent features are afterwards upsampled and expandended back to the original space.
The reconstruction loss is simply the similarity between input and output.

The main idea is to loss every information about the speaker identity and then provide this information after the bottleneck. Providing the identity of a different speaker should thus result in voice conversion.


## Usage

easy usage do this....

For a more detailed explanation of how each part of the model works see [docs/usage](docs/usage_old.md)

## Ethical usage

This package should not be used for harmful or illegal purposes. To avoid this, please make sure that the voices used for conversations can be used under a legal copyright.

## Documentation

For information about the structure of the repo, how to cite and how to use have a look at the `docs/` folder.

