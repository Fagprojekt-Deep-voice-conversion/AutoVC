"""
A class for handling audio data
"""

import librosa
import soundfile as sf
from autovc.audio.tools import remove_noise
import tools
import spectrogram
import inspect 

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
    
    # def get_melspectrogram(model):
    #     pass

    # def get_mel_frames():
    #     pass

    def preprocess(self, pipe_type, *pipeline, **pipeline_args):
        """
        preprocess wav and add to a key in the dictionary with preprocessed versions

        Parameters
        ---------
        pipe_type:
            What to use the pipe line for.
            If eg. 'train' the audio is passed through the pipe line and saved with the key 'train' in the versions.
        *pipeline:
            Names of functions as str in either audio.tools or audio.spectrogram to use on self.wav.
        **pipeline:
            Used to pass arguments for the functions specified in pipeline
            Key is a function in either audio.tools or audio.spectrogram.
            The value is passed as arguments to the specified function
        Return
        ------
        wav:
            The preprocessed wav. this also stored in the versions dict with pipe_type as key
        """
  
        for fun in pipeline:
            func = tools.__dict__.get(fun, spectrogram.__dict__.get(fun, False))
            if not func:
                raise ModuleNotFoundError(f"The function {fun} was not found.")

            # apply function
            args = pipeline_args.get(fun)
            if isinstance(args, dict):
                self.wav = func(self.wav, **args) if "sr" not in inspect.getfullargspec(func).args else func(self.wav, sr = self.sr, **args)
            elif isinstance(args,list):
                self.wav = func(self.wav, *args) if "sr" not in inspect.getfullargspec(func).args else func(self.wav, *args, sr = self.sr, )
            elif args is not None:
                self.wav = func(self.wav, args) if "sr" not in inspect.getfullargspec(func).args else func(self.wav, args, sr = self.sr)
            else:
                self.wav = func(self.wav) if "sr" not in inspect.getfullargspec(func).args else func(self.wav, sr = self.sr)

            
        
        self.versions[pipe_type] = self.wav.copy()
        return self.wav

            


if __name__ == "__main__":
    wav = Audio("data/samples/chooped7.wav")
    # print(wav.sr, wav.versions)
    wav.preprocess("train", "remove_noise")
    wav.save()