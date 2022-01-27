"""
A class for handling audio data
"""

import librosa
import numpy as np
import soundfile as sf
from autovc.audio import tools
from autovc.audio import spectrogram
import inspect 

class Audio:
    def __init__(self, wav, sr = None, sr_org = None) -> None:
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
            self.wav = wav
            self.sr = sr_org
        
        # make dictionary for storing already preprocessed versions of the wav and its sampling rate
        # self.versions = {"org" : [self.wav.copy(), self.sr]}

        # resample audio to target sr
        if sr is not None:
            self.resample(sr)
            # self.sr = sr # this is now the new sampling rate
            # self.versions["resampled"] = [self.wav.copy(), self.sr]
        
        

    def save(self, save_path = "example_audio.wav"):
        """
        Saves the audio data to the given path.
        """
        sf.write(save_path, self.wav, samplerate=self.sr)
        

    def resample(self, sr):
        """
        Resamples self.wav with new sample rate
        """
    
        if sr != self.sr:
            self.wav = librosa.resample(self.wav, self.sr, sr)
            self.sr = sr
        
        return self

    def preprocess(self, *pipeline, **kwargs):
        """
        preprocess wav and add to a key in the dictionary with preprocessed versions

        Parameters
        ---------
        *pipeline:
            Names of functions as str in either audio.tools to use on self.wav.
        **kwargs:
            Kwargs matching a valid key word argument in the function will be given to the function.
            E.g. if two functions are found in the pipeline and the both have have 'arg' as a valid argument and it is found in kwargs, both functions will receive the same value of the argument.
        Return
        ------
        self:
            An instance of the Audio class where the wav has been preprocessed
        """

        if "trim_long_silences" in pipeline:
            possible_srs = np.array([8000, 16000, 32000, 48000])
            sr = possible_srs[np.argmin(abs(possible_srs - self.sr))]
            self.resample(sr)
  
        for fun in pipeline:
            # get function from tools
            if fun is None:
                continue
            func = tools.__dict__.get(fun, False)
            if not func:
                raise ModuleNotFoundError(f"The function {fun} was not found in audio.tools")

            # find kwargs to pass to func
            func_kwargs = {key:value for key, value in kwargs.items() if key in func.__allowed_kw__ + func.__allowed_args__}
            func_kwargs.update({"sr" : self.sr} if "sr" in func.__allowed_kw__ + func.__allowed_args__ else {})
    
            # apply function to wav
            self.wav = func(wav = self.wav, **func_kwargs)
        return self

            


if __name__ == "__main__":
    wav = Audio("data/samples/chooped7.wav")
    # print(wav.sr, wav.versions)
    wav.preprocess("remove_noise", "normalize_volume", time_constant_s = 1)
    wav.save()