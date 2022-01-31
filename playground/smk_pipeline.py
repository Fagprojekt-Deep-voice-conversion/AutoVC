from autovc.audio.tools import combine_audio
from autovc import Audio
from autovc.models import load_model

# data = Audio(combine_audio("data/HYLydspor", sr = 32000), sr_org = 32000, sr = 32000)
# data = data.preprocess( "trim_long_silences", "normalize_volume")
# data.save("data/train/SMK_HY_long.wav")


# data = Audio(combine_audio("artifacts/smk_speakers:v0/HaegueYang_10sek", sr = 32000), sr_org = 32000, sr = 32000)
# data = data.preprocess( "trim_long_silences", "normalize_volume")
# data.save("data/train/yang_long.wav")


# data = Audio(combine_audio("artifacts/smk_speakers:v0/hyang_smk", sr = 32000), sr_org = 32000, sr = 32000)
# data = data.preprocess( "trim_long_silences", "normalize_volume")
# data.save("data/train/yang_long_smk.wav")


# data = Audio(combine_audio("artifacts/smk_speakers:v0/hilde_7sek", sr = 32000), sr_org = 32000, sr = 32000)
# data = data.preprocess( "trim_long_silences", "normalize_volume")
# data.save("data/train/hilde_long.wav")




SE = load_model("speaker_encoder", "SpeakerEncoder_SMK2.pt")
# SE.learn_speaker("louise", "data/train/SMK_HY_louise.wav")
# SE.learn_speaker("yang", "data/train/yang_long_smk.wav")
# SE.learn_speaker("yangYT", "data/train/yangYT_long.wav")
SE.learn_speaker("hilde", "data/train/hilde_long.wav")
SE.save("SpeakerEncoder_SMK2.pt", step = 0)