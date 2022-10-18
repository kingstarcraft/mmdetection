import json
import os
import tqdm
from mmdet.apis import inference_detector, init_detector

#input_root = '//192.168.80.53/amax/cy/color_transfer'
input_root = '//192.168.80.53/amax/cy/color_transfer'
output_root = '//192.168.80.53/amax/longxi/color_transfer_best'
model_root = '//192.168.80.53/amax/longxi/color_transfer/train/train/22_colortransfer'
ground_truth = '//192.168.80.53/home/amax/project/mmdetection/work/mitosis/gt'

target_model = {
    'midog1': {
        'gsnew1': {
            'train23': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_newgs1_va/best_0_bbox_mAP_epoch_405.pth'),
            'train13': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_newgs1_vb/best_0_bbox_mAP_epoch_734.pth'),
            'train12': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_newgs1_vc/best_0_bbox_mAP_epoch_330.pth')
        },
        'he1': {
            'train23': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he1_va/best_0_bbox_mAP_epoch_711.pth'),
            'train13': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he1_vb/best_0_bbox_mAP_epoch_438.pth'),
            'train12': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he1_vc/best_0_bbox_mAP_epoch_425.pth')
        }
    },
    'midog2': {
        'gsnew2': {
            'train23': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_newgs2_va/best_0_bbox_mAP_epoch_985.pth'),
            'train13': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_newgs2_vb/best_0_bbox_mAP_epoch_533.pth'),
            'train12': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_newgs2_vc/best_0_bbox_mAP_epoch_723.pth')
        },
        'he2': {
            'train23': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he2_va/best_0_bbox_mAP_epoch_617.pth'),
            'train13': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he2_vb/best_0_bbox_mAP_epoch_759.pth'),
            'train12': (
                f'F:/111/config.py',
                f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he2_vc/best_0_bbox_mAP_epoch_418.pth')
        }
    }
    #   'midog3': {
    #       'he3': {
    #           'train23': (
    #               f'F:/111/config.py',
    #               f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he3_va/best_0_bbox_mAP_epoch_623.pth'),
    #           'train13': (
    #               f'F:/111/config.py',
    #               f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he3_vb/best_0_bbox_mAP_epoch_555.pth'),
    #           'train12': (
    #               f'F:/111/config.py',
    #               f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he3_vc/best_0_bbox_mAP_epoch_430.pth')
    #       },
    #       'gsnew2': {
    #           'train23': (
    #               f'F:/111/config.py',
    #               f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he1_va/best_0_bbox_mAP_epoch_711.pth'),
    #           'train13': (
    #               f'F:/111/config.py',
    #               f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he1_vb/best_0_bbox_mAP_epoch_438.pth'),
    #           'train12': (
    #               f'F:/111/config.py',
    #               f'{model_root}/cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_he1_vc/best_0_bbox_mAP_epoch_425.pth')
    #       }
    #    }
}


def create_asap(filepath):
    contexts = {}
    if os.path.exists(filepath):
        data = json.load(open(filepath))
        for box in data['contexts']:
            if box['parent'] in contexts:
                contexts[box['parent']].append(box)
            else:
                contexts[box['parent']] = [box]
    else:
        data = {
            "contexts": [],
            "groups": [],
            "info": {"author": "asap"}
        }
    return data, contexts


for method in ['gsnew/gs', 'test12', 'reinhard', 'he']:
    if os.path.isdir(input_root + '/' + method):
        for target in os.listdir(input_root + '/' + method):
            if target not in target_model:
                continue
            if os.path.isdir(input_root + '/' + method + '/' + target):
                models = target_model[target]

                for mtd, datasets in models.items():
                    for dataset, config in datasets.items():
                        model = init_detector(*config, device=0)
                        for source in os.listdir(input_root + '/' + method + '/' + target):
                            data_key = method + '/' + target + '/' + source
                            model_key = f'{mtd}-{dataset}'
                            source_root = input_root + '/' + data_key
                            target_root = output_root + '/' + model_key + '/' + data_key
                            if not os.path.exists(target_root):
                                os.makedirs(target_root)
                            for filename in tqdm.tqdm(os.listdir(source_root), desc=f'{method} {model_key}:{source}->{target}'):
                                stem, ext = os.path.splitext(filename)
                                if os.path.exists(target_root + '/' + f'{stem}.json'):
                                    continue
                                if ext in ['.png', '.tiff', '.tif', '.jpg']:
                                    asap, contexts = create_asap(f'{ground_truth}/{stem}.json')
                                    group = {'name': 'detection', 'parent': -1}
                                    if group not in asap['groups']:
                                        asap['groups'].append(group)
                                    group_id = asap['groups'].index(group)
                                    contexts[group_id] = []

                                    result = inference_detector(model, f'{source_root}/{filename}')
                                    for label, boxes in enumerate(result):
                                        for box in boxes:
                                            contexts[group_id].append({
                                                "name": ['mitosis', 'non-mitosis'][label],
                                                "parent": group_id,
                                                "points": [
                                                    {"x": box[0], "y": box[1]},
                                                    {"x": box[2], "y": box[1]},
                                                    {"x": box[2], "y": box[3]},
                                                    {"x": box[0], "y": box[3]}
                                                ],
                                                "type": "Rectangle",
                                                'info': f'score={box[-1]}'
                                            })
                                    asap['contexts'] = []
                                    for context in contexts.values():
                                        asap['contexts'] += context
                                    json.dump(asap, open(target_root + '/' + f'{stem}.json', 'w'))
