import torch

from mmcv.runner import BaseModule
from abc import ABCMeta


class Standardization(BaseModule, classmethod=ABCMeta):
    def __init__(self, mean, std, to_rgb=True, *args, **kwargs):
        super(Standardization, self).__init__(*args, **kwargs)
        self._mean = torch.Tensor
