from autovc.utils.hpc import create_submit
import os, sys

project = "Trials"
job_name = "SMK_trial_20220125_long"

# script = "autovc/voice_converter.py"
args = " ".join(param.strip() for param in [
    # "-speaker_encoder SpeakerEncoder_SMK.pt",
    "-mode train",
    "-auto_encoder_params ",
    "-model_type auto_encoder",
    "-data_path data/hilde_subset.wav data/SMK_HY_long.wav data/yang_long.wav ",
    "-kwargs n_epochs=20 model_name=SMK_trial_long_20220126.pt cut=True log_freq=4",
    # wandb
    f"-wandb_params project={project} name={job_name}"
])
# execution_code = ["python " + script.strip() + " " + args]
execution_code = ["python autovc/__main__.py" + " " + args]


# set cluster settings
cluster_settings = {
    "overwrite" : True, 
    "n_cores" : 1, 
    "system_memory" : 10, 
    "walltime" : "08:00", 
    "notifications" : True
}

if "win" in sys.platform:
    os.system(*execution_code)
else:
    submit_file = create_submit(job_name, project, *execution_code, **cluster_settings)
    os.system(f"bsub < {submit_file}") # bsubs the created submition file
    # os.system(*execution_code)
