from autovc.utils.hpc import create_submit
import os, sys

project = "NewSpeaker"
job_name = "Louise"

# script = "autovc/voice_converter.py"
args = " ".join(param.strip() for param in [
    "-mode train",
    "-model_type auto_encoder",
    "-n_epochs 10",
    "-auto_encoder models/AutoVC/AutoVC_SMK.pt",
    "-speaker_encoder models/SpeakerEncoder/SpeakerEncoder_SMK.pt",
    "-auto_encoder_params cut=True speakers=False model_name=Louise2",
    "-data_path data/HY",

    # wandb
    f"-wandb_params project={project} name={job_name}"
])
# execution_code = ["python " + script.strip() + " " + args]
execution_code = ["python autovc/voice_converter.py" + " " + args]


# set cluster settings
cluster_settings = {
    "overwrite" : True, 
    "n_cores" : 1, 
    "system_memory" : 5, 
    "walltime" : "08:00", 
    "notifications" : True
}

if "win" in sys.platform:
    os.system(*execution_code)
else:
    submit_file = create_submit(job_name, project, *execution_code, **cluster_settings)
    os.system(f"bsub < {submit_file}") # bsubs the created submition file
    # os.system(*execution_code)
