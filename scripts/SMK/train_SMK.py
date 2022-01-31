from autovc.utils.hpc import create_submit
import os, sys

project = "Trials3"
job_name = "Hilde"

# script = "autovc/voice_converter.py"
args = " ".join(param.strip() for param in [
    # "-speaker_encoder S   peakerEncoder_SMK.pt",
    "-mode train",
    "-auto_encoder_params",
    "-model_type auto_encoder",
    # "-batch_size 16 ",
    # "-auto_encoder autovc_origin.pt",
    "-auto_encoder AutoVC_seed40_200k.pt",
    "-speaker_encoder SpeakerEncoder_SMK2.pt "
    "-data_path data/train/hilde_long.wav data/train/yang_long_smk.wav ",
    "-kwargs n_epochs=256 model_name=SMK_train2.pt cut=True log_freq=32 one_hot=False use_mean_speaker_embedding=True partial_utterance_n_frames=150",
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
    "walltime" : "24:00", 
    "notifications" : True
}

if "win" in sys.platform:
    os.system(*execution_code)
else:
    submit_file = create_submit(job_name, project, *execution_code, **cluster_settings)
    os.system(f"bsub < {submit_file}") # bsubs the created submition file
    # os.system(*execution_code)
