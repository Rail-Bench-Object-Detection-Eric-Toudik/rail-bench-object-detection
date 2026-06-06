import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from models.EdgeCrafter.ecdetseg.engine.edgecrafter.ecvit import ViTAdapter
from models.EdgeCrafter.ecdetseg.engine.edgecrafter.hybrid_encoder import HybridEncoder
from models.EdgeCrafter.ecdetseg.engine.edgecrafter.decoder import ECTransformer
from models.EdgeCrafter.ecdetseg.engine.edgecrafter.postprocessor import PostProcessor

from enum import Enum


class ECDetModel(str, Enum):
    SMALL = "Intellindust/ECDet_S"
    MEDIUM = "Intellindust/ECDet_M"
    LARGE = "Intellindust/ECDet_L"
    XLARGE = "Intellindust/ECDet_X"


class ECDet(nn.Module, PyTorchModelHubMixin):
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


def get_model(config: dict, model: ECDetModel = ECDetModel.MEDIUM):
    """
    Build ECDet model and load pretrained weights.

    Parameters
    ----------
    config : dict
        Model configuration dictionary.
    model : ECDetModel
        Enum selecting which pretrained ECDet variant to load.

    Returns
    -------
    ECDet
        The initialized and pretrained ECDet model.
    """
    return ECDet.from_pretrained(model.value, config=config)
