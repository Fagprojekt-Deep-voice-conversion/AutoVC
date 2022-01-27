from autovc.utils.hpc import create_submit
import os, sys

project = "ForMorten"
job_name = "OrgYangHY"

args = " ".join(param.strip() for param in [
    "-mode train",
    "-model_type auto_encoder",
    "-n_epochs 50",
    "-data_path data/SMK_speakers/HaegueYang_10sek data/HY",
    "-cut True",
    "-preprocess trim_long_silences",
    # wandb
    f"-wandb_params project={project} name={job_name}"
])

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
else: #Only one of the code snippets below should be uncommented
    # uncomment this to create batch job and submit
    submit_file = create_submit(job_name, project, *execution_code, **cluster_settings)
    os.system(f"bsub < {submit_file}") # bsubs the created submition file
    
    # this excutes in terminal
    # os.system(*execution_code)