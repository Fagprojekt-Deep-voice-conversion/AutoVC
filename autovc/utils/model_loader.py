from pathlib import Path
from autovc.speaker_encoder.model import SpeakerEncoder
from autovc.wavernn.model import WaveRNN
from autovc import AutoEncoder
import torch


_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_type: str, model_path: Path, device=None, **model_params):
    """
    Loads a  model
    
    :param model_type: the type of model, can be one of 'auto_encoder', 'speaker_encoder' or 'vocoder'
    :param model_path: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    :**kwargs model_params: various model params can be given, though certain params haveto match the original model
    """

    assert model_type in ['auto_encoder', 'speaker_encoder', 'vocoder']
    device = torch.device(model_params.pop("device", _device))

    if model_type == "auto_encoder":
        model = AutoEncoder(device = device, **model_params)
    elif model_type == "speaker_encoder":
        model = SpeakerEncoder(device = device, **model_params)
    elif model_type == "vocoder":
        model = WaveRNN(device = device, **model_params)

    model.load(weights_fpath=model_path, device = device)
    
    return model

def load_models(model_types, model_paths, params = None, device = None):
    """
    return multiples models

    :param model_types: list of model types
    :param model_paths: list of model_paths
    :param params: list of dictionaries with params to use when initialising model
    """
    params = [{}]*len(model_types) if params == None else params
    models = []
    for model_type, model_path, params in [*zip(model_types, model_paths, params)]:
        if device is not None: params['device'] = device
        models.append(load_model(model_type, model_path, **params))
        # models.append(load_model(model_type, model_path, device))

    return models

