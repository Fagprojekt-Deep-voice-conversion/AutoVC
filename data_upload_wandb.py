import wandb

# to_upload = ["political_speakers", "samples", "SMK_speakers"]
to_upload = ["SMK_speakers"]

for name in to_upload:
    with wandb.init(entity = "deep_voice_inc", project = "data", name = name) as run:
        artifact = wandb.Artifact(name.lower(), "dataset")
        artifact.add_dir("data/" + name)
        run.log_artifact(artifact)