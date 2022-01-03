import wandb
run = wandb.init()
artifact = run.use_artifact('deep_voice_inc/AutoVC/parallel-artifact:v1', type='dataset')
artifact_dir = artifact.download()