import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from ImageReward import ImageReward
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'


_MODELS = {
    "FaceScore": "https://hf-mirror.com/AIGCer-OPPO/FaceScore/blob/main/FS_model.pt",
}


def available_models() -> List[str]:
    """Returns the names of available FS models"""
    return list(_MODELS.keys())


def FaceScore_download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    hf_hub_download(repo_id="AIGCer-OPPO/FaceScore", filename=filename, local_dir=root)
    return download_target


def load(name: str = "FaceScore", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, med_config: str = None):
    """Load a FaceScore model

    Parameters
    ----------
    name : str
        A model name listed by `FaceScore.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/FaceScore"

    Returns
    -------
    model : torch.nn.Module
        The FaceScore model
    """
    if name in _MODELS:
        model_path = FaceScore_download(_MODELS[name], download_root or os.path.expanduser("~/.cache/FaceScore"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    print('load checkpoint from %s'%model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    
    # med_config
    if med_config is None:
        med_config = FaceScore_download("https://huggingface.co/AIGCer-OPPO/FaceScore/resolve/main/med_config.json", download_root or os.path.expanduser("~/.cache/FaceScore"))
    
    model = ImageReward(device=device, med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict,strict=False)
    print("checkpoint loaded")
    model.eval()

    return model




def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target