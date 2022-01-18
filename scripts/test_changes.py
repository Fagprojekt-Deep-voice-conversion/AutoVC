"""Script for performing a small training and conversion to test that the models work"""


from autovc import VoiceConverter

vc = VoiceConverter(wandb_params = {"mode" : "disabled"})

print("Testing Auto Encoder training...")
vc.train(n_epochs = 1, data_path = ["data/samples/mette_183.wav", "data/samples/chooped7.wav"], conversion_examples = False)

print("Testing conversion...")
vc.convert("data/samples/mette_183.wav", "data/samples/chooped7.wav", out_name = "conversion_test.wav")