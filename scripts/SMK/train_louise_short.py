from autovc.utils.hpc import create_submit
import os, sys

project = "Trials"
job_name = "SMK_trial_20220127_short"

# script = "autovc/voice_converter.py"
args = " ".join(param.strip() for param in [
    # "-speaker_encoder SpeakerEncoder_SMK.pt",
    "-mode train",
    "-model_type auto_encoder",
    "-data_path data/newest_trial ",
    "-kwargs n_epochs=50 model_name=SMK_trial_short_20220127.pt data_path_excluded=data/newest_trial/hilde shuffle=True",

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
    "walltime" : "12:00", 
    "notifications" : True
}

if "win" in sys.platform:
    os.system(*execution_code)
else:
    submit_file = create_submit(job_name, project, *execution_code, **cluster_settings)
    os.system(f"bsub < {submit_file}") # bsubs the created submition file
    # os.system(*execution_code)
