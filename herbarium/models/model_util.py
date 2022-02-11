"""Common utilities for building models."""
import logging

import torch
import torchvision

from .herbarium_cnn_model import HerbariumCnnModel
from .herbarium_full_model import HerbariumFullModel
from .herbarium_model import HerbariumModel
from .herbarium_no_orders_model import HerbariumNoOrdersModel
from .hydra_model import HydraModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

MODELS = {
    "herbarium": HerbariumModel,
    "train_all": HerbariumFullModel,
    "no_orders": HerbariumNoOrdersModel,
    "cnn_head": HerbariumCnnModel,
    "hydra": HydraModel,
}

BACKBONES = {
    "b0": {
        "backbone": torchvision.models.efficientnet_b0,
        "size": (224, 224),
        "dropout": 0.2,
        "in_feat": 1280,
    },
    "b1": {
        "backbone": torchvision.models.efficientnet_b1,
        "size": (240, 240),
        "dropout": 0.2,
        "in_feat": 1280,
    },
    "b2": {
        "backbone": torchvision.models.efficientnet_b2,
        "size": (260, 260),
        "dropout": 0.3,
        "in_feat": 1408,
    },
    "b3": {
        "backbone": torchvision.models.efficientnet_b3,
        "size": (300, 300),
        "dropout": 0.3,
        "in_feat": 1536,
    },
    "b4": {
        "backbone": torchvision.models.efficientnet_b4,
        "size": (380, 380),
        "dropout": 0.4,
        "in_feat": 1792,
    },
    # b5: {"size": (456, 456), }
    # b6: {"size": (528, 528), }
    "b7": {
        "backbone": torchvision.models.efficientnet_b7,
        "size": (600, 600),
        "dropout": 0.5,
        "in_feat": 2560,
    },
}


def get_backbone_params(model, backbone):
    """Setup model params gotten from the backbone params."""
    model_params = BACKBONES[backbone]
    model.size = model_params["size"]
    model.mean = model_params.get("mean", IMAGENET_MEAN)
    model.std_dev = model_params.get("std_dev", IMAGENET_STD_DEV)


def load_model_state(model, load_model):
    """Load a previous model."""
    model.state = torch.load(load_model) if load_model else {}
    if model.state.get("model_state"):
        logging.info("Loading a model.")
        model.load_state_dict(model.state["model_state"])
