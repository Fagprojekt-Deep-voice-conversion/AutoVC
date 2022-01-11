# import wandb
# run = wandb.init()
# artifact = run.use_artifact('deep_voice_inc/AutoVC/parallel-artifact:v1', type='dataset')
# artifact_dir = artifact.download()


# from autovc.utils.hparams import WaveRNNParams
import wandb
# hparams = dict(yaml = "playground/config_test.yaml")
import yaml
with open("playground/config_test.yaml", 'r') as stream:
    hparams = yaml.safe_load(stream)
# print(hparams)

wandb.init(config=hparams, mode = "disabled")

print(wandb.config.epochs)