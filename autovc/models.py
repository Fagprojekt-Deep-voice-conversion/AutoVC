from autovc.speaker_encoder import SpeakerEncoder
from autovc.auto_encoder import AutoEncoder
from autovc.wavernn.model import WaveRNN
import torch
from autovc.utils.hparams import WaveRNNParams, AutoEncoderParams, SpeakerEncoderParams

_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_type: str, model_name, model_dir = None, device=None, **model_params):
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
        model_dir = AutoEncoderParams["model_dir"] if model_dir is None else model_dir
    elif model_type == "speaker_encoder":
        model = SpeakerEncoder(device = device, **model_params)
        model_dir = SpeakerEncoderParams["model_dir"] if model_dir is None else model_dir
    elif model_type == "vocoder":
        model = WaveRNN(device = device, **model_params) 
        model_dir = WaveRNNParams["model_dir"] if model_dir is None else model_dir

    model.load(model_name=model_name, model_dir=model_dir, device = device)
    
    return model

def load_models(model_types, model_names, model_dirs = [None]*3, params = None, device = None):
    """
    return multiples models

    :param model_types: list of model types
    :param model_paths: list of model_paths
    :param params: list of dictionaries with params to use when initialising model
    """
    params = [{}]*len(model_types) if params == None else params
    models = []
    for model_type, model_name, model_dir, param in [*zip(model_types, model_names, model_dirs, params)]:
        if device is not None: 
            param['device'] = device
        models.append(load_model(model_type, model_name, model_dir, **param))
        # models.append(load_model(model_type, model_path, device))

    return models

