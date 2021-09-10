from abc import ABCMeta
from collections import OrderedDict

import torch
import torch.distributed as dist

from mmcv.runner import BaseModule, auto_fp16
from ..builder import DETECTORS, build_backbone, build_loss, build_detector


@DETECTORS.register_module()
class GAN(BaseModule, metaclass=ABCMeta):
    def __init__(self, generator, discriminator,
                 generator_loss=dict(type='MSELoss', loss_weight=0.01),
                 discriminator_loss=dict(type='MSELoss', loss_weight=1), **kwargs):
        super(GAN, self).__init__()
        self.generator = build_detector(generator, **kwargs)
        if isinstance(discriminator, dict):
            discriminator = [discriminator]
        if isinstance(discriminator, (tuple, list)):
            discriminate = {}
            for param in discriminator:
                index = param.pop('index')
                shared = param.pop('shared', False)
                if isinstance(index, int):
                    discriminate[f'discriminator{index}'] = build_backbone(param)
                elif isinstance(index, (tuple, list)):
                    feature = build_backbone(param) if shared else None
                    for i in index:
                        discriminate[f'discriminator{i}'] = feature if shared else build_backbone(param)
                else:
                    raise NotImplementedError
            self.discriminator = torch.nn.ModuleDict(discriminate)
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.generator_loss = build_loss(generator_loss)
        self.discriminator_loss = build_loss(discriminator_loss)

    def generate(self, imgs):
        return self.generator.extract_feat(imgs)

    def discriminate(self, real, fake):
        return [r.detach() for r in real], [f.detach() for f in fake]

    def parameters(self, recurse: bool = True):
        generator = self.generator.parameters(recurse)
        discriminator = self.discriminator.parameters(recurse)
        if self.training:
            return {'generator': generator, 'discriminator': discriminator}
        else:
            return super(GAN, self).parameters(recurse)

    def forward_train(self, imgs, img_metas, fakes, **kwargs):
        real_features, losses = self.generator.forward_train(imgs, img_metas, return_feature=True, **kwargs)
        fake_features = self.generate(fakes)

        if not isinstance(real_features, (tuple, list)):
            real_features = [real_features]
        if not isinstance(fake_features, (tuple, list)):
            fake_features = [fake_features]

        generate_losses = []
        for key, model in self.discriminator.items():
            index = int(key.replace('discriminator', ''))
            generator_predict = model(fake_features[index])
            generate_loss = self.generator_loss(generator_predict, self.real_label.expand_as(generator_predict))
            generate_losses.append(generate_loss)

        real_features, fake_features = self.discriminate(real_features, fake_features)
        discriminate_losses = []
        for key, model in self.discriminator.items():
            index = int(key.replace('discriminator', ''))
            real_predict = model(real_features[index])
            fake_predict = model(fake_features[index])
            real_loss = self.discriminator_loss(real_predict, self.real_label.expand_as(real_predict))
            fake_loss = self.discriminator_loss(fake_predict, self.real_label.expand_as(fake_predict))
            discriminate_losses.append(0.5 * (real_loss + fake_loss))
        generate_loss = torch.mean(torch.stack(generate_losses))
        discriminate_loss = torch.mean(torch.stack(discriminate_losses))
        losses.update({'generate_loss': generate_loss, 'discriminate_loss': discriminate_loss})
        return losses

    async def async_simple_test(self, img, img_metas, **kwargs):
        return self.generator.async_simple_test(img, img_metas, **kwargs)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        return self.generator.aforward_test(img, img_metas, **kwargs)

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.generator.forward_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, fake=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, fake, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        generate_loss = log_vars.pop('generate_loss')
        discriminate_loss = log_vars.pop('discriminate_loss')
        detector_loss = sum(_value for _key, _value in log_vars.items()
                            if 'loss' in _key)

        log_vars['detector_loss'] = detector_loss
        log_vars['generate_loss'] = generate_loss
        log_vars['discriminate_loss'] = discriminate_loss
        log_vars['loss'] = generate_loss + detector_loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        generate_loss = log_vars.pop('loss')
        return generate_loss, discriminate_loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        generate_loss, discriminate_loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss={'generator': generate_loss, 'discriminator': discriminate_loss},
            log_vars=log_vars,
            num_samples=len(data['img_metas'])
        )

        return outputs

    def val_step(self, data, optimizer=None):
        return self.generator.val_step(data, optimizer)

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        return self.generator.show_result(img,
                                          result,
                                          score_thr,
                                          bbox_color,
                                          text_color,
                                          mask_color,
                                          thickness,
                                          font_size,
                                          win_name,
                                          show,
                                          wait_time,
                                          out_file)

    def onnx_export(self, img, img_metas):
        return self.generator.onnx_export(img, img_metas)
