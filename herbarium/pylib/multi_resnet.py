"""Override RestNet so that it uses multiple inputs on the forward pass."""
# import torch
# import torchvision
# from torch import nn
# from torch import Tensor
#
#
# class MultiResNet(nn.Module):
#     """Override ResNet so that it uses multiple inputs on the forward pass."""
#
#     def __init__(self, backbone, orders_len, load_weights, freeze, out_features=1):
#         super().__init__()
