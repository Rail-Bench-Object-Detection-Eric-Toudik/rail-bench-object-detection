import os
import torch
import torch.nn as nn

from models.EdgeCrafter.ecdetseg.engine.edgecrafter.ecvit import ViTAdapter
from models.EdgeCrafter.ecdetseg.engine.edgecrafter.hybrid_encoder import HybridEncoder
from models.EdgeCrafter.ecdetseg.engine.edgecrafter.decoder import ECTransformer
from models.EdgeCrafter.ecdetseg.engine.edgecrafter.postprocessor import PostProcessor
from read_config import read_config

from enum import Enum


class ECDetModel(str, Enum):
    SMALL = "Intellindust/ECDet_S"
    MEDIUM = "Intellindust/ECDet_M"
    LARGE = "Intellindust/ECDet_L"
    XLARGE = "Intellindust/ECDet_X"


class ECDet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = ViTAdapter(
            **config["backbone"]
        )
        self.encoder = HybridEncoder(
            **config["encoder"]
        )
        self.decoder = ECTransformer(
            **config["decoder"]
        )
        self.postprocessor = PostProcessor(
            **config["postprocessor"]
        )

    def forward(self, x, orig_target_sizes):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.postprocessor(x, orig_target_sizes)

        return x


def get_model(weights: str, model_size: ECDetModel = ECDetModel.MEDIUM, map_location: str = "cpu", strict: bool = True):
    """
    Build an ECDet PyTorch model and load local ``.pth`` weights.

    Parameters
    ----------
    weights : str
        Path to a local ``.pth`` checkpoint file.
    model_size : ECDetModel, default=ECDetModel.MEDIUM
        Model variant selector. Accepts ``ECDetModel`` enum entry
        (e.g. ``ECDetModel.MEDIUM.value``).
    map_location : str, default="cpu"
        Device can be ``"cpu"`` or a specific CUDA device (e.g. ``"cuda:0"``).
        Device mapping used while loading the checkpoint.
    strict : bool, default=True
        Strictness forwarded to ``load_state_dict``.

    Returns
    -------
    ECDet
        Initialized ECDet model with loaded checkpoint weights.

    Raises
    ------
    FileNotFoundError
        If ``weights`` does not exist.
    ValueError
        If ``model_value`` is invalid.
    """
    config_map = {
        ECDetModel.SMALL.value: "ecdet_small_config.yaml",
        ECDetModel.MEDIUM.value: "ecdet_medium_config.yaml",
        ECDetModel.LARGE.value: "ecdet_large_config.yaml",
        ECDetModel.XLARGE.value: "ecdet_xlarge_config.yaml",
    }

    selected_model_size = model_size.value if isinstance(model_size, ECDetModel) else model_size
    config_filename = config_map.get(selected_model_size)
    if not config_filename:
        raise ValueError(f"Invalid model value: {model_size}")

    project_root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(project_root, "config", config_filename)

    config = read_config(config_path)
    ecdet = ECDet(config)

    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights file not found: {weights}")

    checkpoint = torch.load(weights, map_location=map_location)

    if isinstance(checkpoint, dict) and 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format in: {weights}")

    # Allow checkpoints saved from DDP wrappers.
    state_dict = {
        (key[7:] if key.startswith('module.') else key): value
        for key, value in state_dict.items()
    }

    ecdet.load_state_dict(state_dict, strict=strict)
    return ecdet