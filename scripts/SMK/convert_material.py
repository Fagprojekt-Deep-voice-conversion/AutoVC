from autovc.utils.hpc import create_submit
import os, sys

# script = "autovc/voice_converter.py"
args = " ".join(param.strip() for param in [
    "-mode convert",
    "-auto_encoder models/AutoVC/model_20220119RU35A9.pt",
    "-speaker_encoder models/SpeakerEncoder/SpeakerEncoder_SMK.pt",
    # "-auto_encoder_params cut=True speakers=True",
    "-sources data/HY -targets HaegueYang",
    # "-convert_params pipes={output:[normalize_volume,remove_noise],source:[normalize_volume]}",
    "-convert_params out_dir=SMK_material"
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
