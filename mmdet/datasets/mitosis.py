from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class MitosisDataset(CocoDataset):

    CLASSES = ('mitosis', 'non-mitosis')
