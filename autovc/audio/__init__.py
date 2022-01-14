"""
A class for handling audio data
"""

import librosa
import soundfile as sf

class Audio:
    def __init__(self, wav, sr = 22050, sr_org = None) -> None:
        """
        Initlialises the audio class

        Parameters
        ----------
        wav:
            a path to a wav file or a np array with audio data
        sr:
            the samplerate to give the audio data (librosa.resample is used for this)
            if None, no resampling is done
        sr_org:
            the samplerate of the given audio data, this is only necesary if wav is not a path
        """

        

        # load file
        if isinstance(wav, str):
            self.wav_path = wav
            # self.wav, self.sr_org = librosa.load(self.wav_path, sr=sr_org)
            self.wav, self.sr = librosa.load(self.wav_path, sr=sr_org)
        else:
            assert sr_org is not None, "sr_org must be given if wav is not a file path"
            self.wav_path = None
            # self.sr_org = sr_org
            self.wav = wav
        
        # make dictionary for storing already preprocessed versions of the wav and its sampling rate
        self.versions = {"org" : [self.wav.copy(), self.sr]}

        # resample audio to target sr
        # self.wav_org = self.wav.copy() # store orginal wav (to use if wav has been changed for the worse)
        # self.sr = self.sr_org # sr is set to origninal sampling rate
        if sr is not None:
            self.wav = self.resample(sr)
            self.sr = sr # this is now the new sampling rate
            self.versions["resampled"] = [self.wav.copy(), self.sr]
        
        

    def save(self, save_path = "example_audio.wav", sr = None):
        """
        Saves the audio data to the given path with the given sampling rate (sr). Sampling rate is set objects value if None is given
        """
        sr = self.sr if sr is None else sr
        sf.write(save_path, self.resample(sr), samplerate=sr)
        

    def resample(self, sr):
        """
        Resamples self.wav with new sample rate
        """
        if sr == self.sr:
            return self.wav
        else:
            return librosa.resample(self.wav, self.sr, sr)
    
    def get_mels(model):
        pass

    def preprocess(name, *pipeline):
        """
        preprocess wav and add to a key in the dictionary with preprocessed versions
        """
        pass


if __name__ == "__main__":
    wav = Audio("data/samples/chooped7.wav")
    print(wav.sr, wav.versions)
    wav.save()