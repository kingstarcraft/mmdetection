import numpy as np

import torch
import zero
from mmcv.runner import BaseModule
from abc import ABCMeta

from ..builder import PIPELINES


@PIPELINES.register_module()
class Standardization(BaseModule, metaclass=ABCMeta):
    def __init__(self, mean, std, to_rgb=True):
        super(Standardization, self).__init__()
        self._info = dict(mean=mean, std=std, to_rgb=to_rgb)
        self._mean = torch.nn.Parameter(torch.tensor(mean)[..., None, None], requires_grad=False)
        self._std = torch.nn.Parameter(torch.tensor(std)[..., None, None], requires_grad=False)
        self._to_rgb = to_rgb

    def forward(self, img, img_metas, **kwargs):
        if self._to_rgb:
            if img.shape[-3] == 3:
                img = img[..., [2, 1, 0], :, :]
            elif img.shape[-3] == 4:
                img = img[..., [2, 1, 0, 3], :, :]

        img = (img - self._mean) / self._std
        for meta in img_metas:
            meta['img_norm_cfg'] = self._info
        return img, img_metas, kwargs


@PIPELINES.register_module()
class QualityDistribution(BaseModule, metaclass=ABCMeta):
    def __init__(self, domains, limit=None, mean=False):
        super(QualityDistribution, self).__init__()
        density, const = [], []
        for domain in domains:
            density.append(domain['density'])
            const.append(domain['const'])

        self._domain = []
        for domain in range(len(domains)):
            if limit is not None:
                if domain in limit:
                    continue
            self._domain.append(domain)
        self._density = np.array(density)
        self._const = np.array(const)
        self._net = zero.torch.net.quality.Transform()
        self._mean = mean

    def forward(self, img, img_metas, **kwargs):
        start_index = kwargs.get('domain').cpu().numpy()
        end_index = np.random.choice(self._domain, len(start_index))
        density_start = self._density[start_index]
        density_end = self._density[end_index]
        density_alpha = np.random.uniform(0, 1, len(start_index))
        density_target = zero.torch.net.quality.solve(density_start, density_end, density_alpha)
        if self._mean:
            const_start = self._const[start_index]
            const_end = self._const[end_index]
            const_alpha = np.random.uniform(0, 1, (len(start_index), 1))
            const = const_start * (1 - const_alpha) + const_alpha * const_end
        else:
            const = None

        return self._net(img, density_start, density_target, const), img_metas, kwargs
