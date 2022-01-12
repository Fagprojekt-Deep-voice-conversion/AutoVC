from autovc.utils.hpc import create_submit
import os

project = "GettingStarted"
job_name = "test"

script = "autovc/voice_converter.py"
args = " ".join(param.strip() for param in [
    "-mode train",
    "-model_type auto_encoder",
    "-n_epochs 1",

    # wandb
    f"-wandb_params project={project} name={job_name} mode=disabled"
])
execution_code = ["python " + script.strip() + " " + args]


# set cluster settings
cluster_settings = {
    "overwrite" : True, 
    "n_cores" : 1, 
    "system_memory" : 5, 
    "walltime" : "04:00", 
    "notifications" : True
}


submit_file = create_submit(job_name, project, *execution_code, **cluster_settings)
os.system(f"bsub < {submit_file}") # bsubs the created submition file
