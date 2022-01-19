"""Script for performing a small training and conversion to test that the models work"""


from autovc import VoiceConverter
from autovc.utils.model_loader import load_model

vc = VoiceConverter(wandb_params = {"mode" : "disabled"})

print("Testing Auto Encoder training...")
vc.train(n_epochs = 1, data_path = ["data/samples/mette_183.wav", "data/samples/chooped7.wav"], conversion_examples = False)

print("Testing conversion...")
vc.convert("data/samples/mette_183.wav", "data/samples/chooped7.wav", out_name = "conversion_test.wav")

print("Testing speaker encoder...")
SE = load_model('speaker_encoder', 'models/SpeakerEncoder/SpeakerEncoder.pt')
# SE.learn_speaker('hilde', 'data/test_data/hilde_7sek')
# SE.learn_speaker('yang', 'data/test_data/HaegueYang_10sek')
# SE.save('SpeakerEncoder2.pt')
# SE = load_model('speaker_encoder', 'SpeakerEncoder2.pt')
SE.learn_speaker("hilde", ["data/samples/hilde_1.wav", "data/samples/hilde_301.wav"])
print(SE.speakers)