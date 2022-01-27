from autovc.utils.hpc import create_submit
import os, sys

project = "test"
job_name = "test_speakers"

# script = "autovc/voice_converter.py"
args = " ".join(param.strip() for param in [
    "-mode train",
    "-mean_speaker_path louise=data/HY",
    "-model_type auto_encoder",
    # "-speaker_encoder SpeakerEncoder_SMK.pt",
    # "-auto_encoder_params cut=True speakers=True",
    # "-data_path data/yang_long.wav data/samples/chooped7.wav",
    "-data_path data/samples",
    "-target_examples louise", 
    "-n_epochs 2",
    "-log_freq 1",

    # wandb
    f"-wandb_params project={project} name={job_name}"
])
# execution_code = ["python " + script.strip() + " " + args]
execution_code = ["python autovc/__main__.py" + " " + args]

# execution_code = ["python autovc/voice_converter.py"]

# set cluster settings
cluster_settings = {
    "overwrite" : True, 
    "n_cores" : 1, 
    "system_memory" : 15, 
    "walltime" : "08:00", 
    "notifications" : True
}

if "win" in sys.platform:
    os.system(*execution_code)
else:
    # submit_file = create_submit(job_name, project, *execution_code, **cluster_settings)
    # os.system(f"bsub < {submit_file}") # bsubs the created submition file
    os.system(*execution_code)
