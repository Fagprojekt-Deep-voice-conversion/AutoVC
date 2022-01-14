# List of current audio functions

Only files in the autovc folder are included

| Function          | Origin script         | Used in       | Short description         | Category      | 
|-------------------|-----------------------|---------------|---------------------------|---------------|
| `normalize()`     | utils/audio.py        | `audio_to_melspectrogram()`       |                           |               |
| `denormalize()`   | utils/audio.py        | Nowhere       |                           |               |
| `amp_to_db()`     | utils/audio.py        | `audio_to_melspectrogram()`              |                           |               |
| `db_to_amp()`     | utils/audio.py        | Nowhere              |                           |               |
| `audio_to_melspectrogram()` | utils/audio.py | `VoiceConverter.convert()`<br>`TrainDataLoader()`| | |
| `preprocess_wav()`| utils/audio.py        |`VoiceConverter.convert()`<br>`TrainDataLoader()`<br>`SpeakerEncoderDataLoader()` | | |
| `create_audio_mask()`     | utils/audio.py        |`split_audio()`<br>`trim_long_silences()`| | |
| `trim_long_silences()`     | utils/audio.py        |`preprocess_wav()` |                           |               |
| `normalize_volume()`     | utils/audio.py        | `preprocess_wav()`|                           |               |
| `split_audio()`     | utils/audio.py        | Used before everything |                           |               |
| `remove_noise()`     | utils/audio.py        | `VoiceConverter.convert()` <br> `preprocess_wav()`|                           |               |
| `combine_audio()`     | utils/audio.py        | Nowhere |                           |               |
| `get_mel_frames()`     | utils/audio.py        | `TrainDataLoader()`<br>`SpeakerEncoderDataLoader()`|                           |               |
| `wav_to_mel_spectrogram()`     | speaker_encoder/utils.py        | `speaker_encoder.model.embed_utterance()`<br> `SpeakerEncoderDataLoader()`              |                           |               |
| `compute_partial_slices()`     | speaker_encoder/utils.py        | `speaker_encoder.model.embed_utterance()`<br>`utils/audio.get_mel_frames()`|                           |               |
