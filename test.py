import wandb
run = wandb.init()
artifact = run.use_artifact('deep_voice_inc/Trials3/SMK_train2.pt:v107', type='AutoEncoder')
artifact_dir = artifact.download("models/AutoVC")