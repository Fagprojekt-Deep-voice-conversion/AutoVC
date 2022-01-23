"""Script for performing a small training and conversion to test that the models work"""


from autovc import VoiceConverter

vc = VoiceConverter(wandb_params = {"mode" : "online"}, speaker_encoder = "../../models/SpeakerEncoder/SpeakerEncoder.pt")

print("Testing Auto Encoder training...")
vc.train(n_epochs = 1, data_path = ["data/samples/mette_183.wav", "data/samples/chooped7.wav"], source_examples = False)
# vc.train(n_epochs = 1, data_path = ["data/yang_long.wav", "data/samples/chooped7.wav"], conversion_examples = False, auto_encoder_params = {"cut" : True, "speakers" : True})

print("Testing conversion...")
vc.convert("data/samples/mette_183.wav", "data/samples/chooped7.wav", save_name = "conversion_test.wav")


print("Testing speaker encoder...")
vc.SE.learn_speaker("yang", ["data/samples/chooped7.wav", "data/samples/HaegueSMK.wav", "data/samples/HaegueYang_5.wav"])
print(vc.SE.speakers)
vc.convert("data/samples/mette_183.wav", "yang", save_name = "conversion_test2.wav", preprocess=[])