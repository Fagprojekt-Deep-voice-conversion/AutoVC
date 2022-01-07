from pathlib import Path
from autovc.speaker_encoder.model import SpeakerEncoder
from autovc.wavernn.model import WaveRNN
from autovc.auto_encoder.model_vc import Generator
import torch


_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_type: str, model_path: Path, device=None):
    """
    Loads a  model
    
    :param model_type: the type of model, can be one of 'auto_encoder', 'speaker_encoder' or 'vocoder'
    :param model_path: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """

    assert model_type in ['auto_encoder', 'speaker_encoder', 'vocoder']
    device = torch.device(device if device is not None else _device)

    if model_type == "auto_encoder":
        model = Generator(device = device)
    elif model_type == "speaker_encoder":
        model = SpeakerEncoder(device = device)
    elif model_type == "vocoder":
        model = WaveRNN(device = device)

    model.load(weights_fpath=model_path, device = device)
    
    return model

def load_models(model_types, model_paths, device = None):
    """
    return multiples models

    :param model_types: list of model types
    :param model_paths: list of model_paths
    """
    models = []
    for model_type, model_path in [*zip(model_types, model_paths)]:
        models.append(load_model(model_type, model_path, device))

    return models

