"""All current models"""
from .base_model import BaseModel
from .cnn_model import CnnModel
from .echo_model import EchoModel
from .full_model import FullModel
from .hydra_model import HydraModel
from .no_orders_model import NoOrdersModel

MODELS = {
    "base": BaseModel,
    "train_all": FullModel,
    "no_orders": NoOrdersModel,
    "cnn_head": CnnModel,
    "echo": EchoModel,
    "hydra": HydraModel,
}
