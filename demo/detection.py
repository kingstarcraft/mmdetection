import json
import os
import tqdm
from mmdet.apis import inference_detector, init_detector

input_root = '//192.168.80.53/amax/cy/color_transfer/__pycache__'
output_root = '//192.168.80.53/amax/longxi/color_transfer'
model_root = '//192.168.80.53/home/amax/project/mmdetection/work/mitosis'
ground_truth = '//192.168.80.53/home/amax/project/mmdetection/work/mitosis/gt'

target_model = {
    'midog1': (
        f'{model_root}/test/midog1.py',
        f'{model_root}/train/20_colortransfer/v20.0.0_cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_a_0.8/best_bbox_mAP_epoch_1116.pth',
    ),
    'midog2': (
        f'{model_root}/test/midog2.py',
        f'{model_root}/train/20_colortransfer/v20.0.0_cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_b_0.8/best_bbox_mAP_epoch_1093.pth',
    ),
    'midog3': (
        f'{model_root}/test/midog3.py',
        f'{model_root}/train/20_colortransfer/v20.0.0_cascade_rcnn_r50_fpn_0.8_0.9_1280_1x_c_0.8/epoch_2216.pth',
    ),
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


for method in ['test12', 'reinhard', 'gs', 'he', 'gsnew/gs']:
    if os.path.isdir(input_root + '/' + method):
        for target in os.listdir(input_root + '/' + method):
            if target not in target_model:
                continue
            if os.path.isdir(input_root + '/' + method + '/' + target):
                model = init_detector(*target_model[target], device=0)
                for source in os.listdir(input_root + '/' + method + '/' + target):
                    source_root = input_root + '/' + method + '/' + target + '/' + source
                    target_root = output_root + '/' + method + '/' + target + '/' + source
                    if not os.path.exists(target_root):
                        os.makedirs(target_root)
                    for filename in tqdm.tqdm(os.listdir(source_root)):
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
