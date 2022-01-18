from autovc.utils.model_loader import load_model

SE = load_model('speaker_encoder', 'artifacts/speaker1/model_20220117.pt')
SE.learn_speaker('hilde', 'data/test_data/hilde_7sek')
SE.learn_speaker('yang', 'data/test_data/HaegueYang_10sek')
SE.save('SpeakerEncoder2.pt')
SE = load_model('speaker_encoder', 'SpeakerEncoder2.pt')
print(SE.speakers)