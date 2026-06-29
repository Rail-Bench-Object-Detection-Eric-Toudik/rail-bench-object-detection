import os
import torch
import torch.nn as nn

from models.EdgeCrafter.ecdetseg.engine.edgecrafter.ecvit import ViTAdapter
from models.EdgeCrafter.ecdetseg.engine.edgecrafter.hybrid_encoder import HybridEncoder
from models.EdgeCrafter.ecdetseg.engine.edgecrafter.decoder import ECTransformer
from models.EdgeCrafter.ecdetseg.engine.edgecrafter.postprocessor import PostProcessor

try:
    from src.read_config import read_config
except ModuleNotFoundError:
    from read_config import read_config

from enum import Enum


class ECDetModel(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


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


def _get_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "ema" in checkpoint and isinstance(checkpoint["ema"], dict):
            return checkpoint["ema"]["module"]
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def get_model(
    model: ECDetModel = ECDetModel.MEDIUM,
    checkpoint_path: str | os.PathLike | None = None,
):
    """
    Build ECDet model and optionally load local checkpoint weights.

    Parameters
    ----------
    model : ECDetModel
        Enum selecting which ECDet variant to build.
    checkpoint_path : str | os.PathLike | None
        Optional path to a local checkpoint downloaded from W&B.

    Returns
    -------
    ECDet
        The initialized ECDet model.
    """
    config_map = {
        ECDetModel.SMALL: "ecdet_small_config.yaml",
        ECDetModel.MEDIUM: "ecdet_medium_config.yaml",
        ECDetModel.LARGE: "ecdet_large_config.yaml",
        ECDetModel.XLARGE: "ecdet_xlarge_config.yaml",
    }

    config_filename = config_map.get(model)
    if not config_filename:
        raise ValueError(f"Invalid model enum value: {model}")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(PROJECT_ROOT, "config", config_filename)

    config = read_config(config_path)

    if "use_encoder_idx" in config["encoder"]:
        config["encoder"]["use_encoder_idx"] = [
            idx for idx in config["encoder"]["use_encoder_idx"] if idx is not None
        ]

    if checkpoint_path is not None:
        config["backbone"]["skip_load_backbone"] = True

    ecdet = ECDet(config)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = _get_checkpoint_state_dict(checkpoint)
        ecdet.load_state_dict(state_dict)

    return ecdet
