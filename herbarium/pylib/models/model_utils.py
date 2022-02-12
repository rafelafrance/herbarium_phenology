"""Common utilities for building models."""
import logging

import torch

from .backbones import BACKBONES
from .backbones import IMAGENET_MEAN
from .backbones import IMAGENET_STD_DEV


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
