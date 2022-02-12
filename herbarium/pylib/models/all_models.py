"""All current models"""
from .herbarium_cnn_model import HerbariumCnnModel
from .herbarium_full_model import HerbariumFullModel
from .herbarium_model import HerbariumModel
from .herbarium_no_orders_model import HerbariumNoOrdersModel
from .hydra_model import HydraModel

MODELS = {
    "utils": HerbariumModel,
    "train_all": HerbariumFullModel,
    "no_orders": HerbariumNoOrdersModel,
    "cnn_head": HerbariumCnnModel,
    "hydra": HydraModel,
}
