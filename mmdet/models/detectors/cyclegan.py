import random
import torch
from torchvision.transforms import CenterCrop

from .gan import GAN
from ..builder import DETECTORS


@DETECTORS.register_module()
class CycleGAN(GAN):
    class Pool(list):
        def __init__(self, size, probability=0.5):
            super(CycleGAN.Pool, self).__init__()
            self._size = size
            self._probability = probability

        def __call__(self, inputs):
            if self._size <= 0:
                return inputs
            outputs = []
            for input in inputs:
                if len(self) < self._size:
                    self.append(input)
                    outputs.append(input)
                else:
                    if random.uniform(0, 1) > self._probability:
                        id = random.randint(0, self._size - 1)
                        outputs.append(self[id].clone())
                        self[id] = input
                    else:
                        outputs.append(input)
            return outputs

    def __init__(self, generator, discriminator,
                 loss=dict(type='MSELoss', loss_weight=0.01),
                 pool=dict(size=50, probability=0.5),
                 **kwargs):
        super(CycleGAN, self).__init__(generator, discriminator, loss, **kwargs)

        self.pools = None
        self.build_pool = lambda: self.Pool(**pool)

    def discriminate(self, real, fake):
        real, fake = super(CycleGAN, self).discriminate(real, fake)
        if self.pools is None:
            self.pools = [self.build_pool() for _ in fake]
        assert len(real) == len(fake) == len(self.pools)
        fake = [pool(f) for pool, f in zip(self.pools, fake)]

        method = max if random.uniform(0, 1) > 0.5 else min
        outputs = [], []

        for r, f in zip(real, fake):
            h = method(r.shape[-2], *[_.shape[-2] for _ in f])
            w = method(r.shape[-1], *[_.shape[-1] for _ in f])
            r = CenterCrop((h, w))(r)
            f = torch.stack([CenterCrop((h, w))(_) for _ in f])
            outputs[0].append(r)
            outputs[1].append(f)
        return outputs
