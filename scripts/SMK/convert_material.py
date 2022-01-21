from autovc.utils.hpc import create_submit
import os, sys

# script = "autovc/voice_converter.py"
args = " ".join(param.strip() for param in [
    "-mode convert",
    # "-auto_encoder models/AutoVC/model_20220121H0M23N.pt",
    # "-auto_encoder models/AutoVC/Louise2",
    "-auto_encoder models/AutoVC/AutoVC_SMK.pt",
    "-speaker_encoder models/SpeakerEncoder/SpeakerEncoder_SMK.pt",
    # "-sources data/HY -targets HaegueYang",
    # "-sources data/HY/HY1.wav -targets HaegueYang",
    "-sources data/splitted_wavs -targets HaegueYang",
    # "-convert_params pipes={source:[normalize_volume],output:[normalize_volume,remove_noise]}",
    # "-convert_params save_dir=HY1 cut=True partial_utterance_n_frames=160"
    "-convert_params save_dir=HY1",
])
# execution_code = ["python " + script.strip() + " " + args]
execution_code = ["python autovc/voice_converter.py" + " " + args]


# set cluster settings
cluster_settings = {
    "overwrite" : True, 
    "n_cores" : 1, 
    "system_memory" : 5, 
    "walltime" : "01:00", 
    "notifications" : True
}

if "win" in sys.platform:
    os.system(*execution_code)
else:
    # submit_file = create_submit(job_name, project, *execution_code, **cluster_settings)
    # os.system(f"bsub < {submit_file}") # bsubs the created submition file
    os.system(*execution_code)
