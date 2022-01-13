from autovc.utils.hpc import create_submit
import os, sys

project = "GettingStarted"
job_name = "test"

# script = "autovc/voice_converter.py"
args = " ".join(param.strip() for param in [
    "-mode train",
    "-model_type auto_encoder",
    "-n_epochs 1",
    # "-data_path data/SMK_train/newest_trial",

    # wandb
    f"-wandb_params project={project} name={job_name}"
])
# execution_code = ["python " + script.strip() + " " + args]
execution_code = ["python autovc" + " " + args]


# set cluster settings
cluster_settings = {
    "overwrite" : True, 
    "n_cores" : 1, 
    "system_memory" : 5, 
    "walltime" : "04:00", 
    "notifications" : True
}

if "win" in sys.platform:
    os.system(*execution_code)
else:
    submit_file = create_submit(job_name, project, *execution_code, **cluster_settings)
    os.system(f"bsub < {submit_file}") # bsubs the created submition file
