# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
import json
from argparse import ArgumentParser

import tqdm

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--list', default=None, help='list of image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


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


def main(args):
    # build the model from a config file and a checkpoint file
    if os.path.isfile(args.checkpoint):
        checkpoints = [args.checkpoint]
    else:
        checkpoints = os.listdir(args.checkpoint)

    for checkpoint in checkpoints:
        stem, ext = os.path.splitext(checkpoint)
        if ext in ['.pth']:
            group = {'name': stem, 'parent': -1}

            model = init_detector(args.config, args.checkpoint + '/' + checkpoint, device=args.device)
            # test a single image
            if os.path.isfile(args.input):
                filenames = [args.input]
                root = ''
            else:
                root = args.input
                if args.list is None:
                    filenames = os.listdir(root)
                elif args.list.endswith('.json'):
                    filenames = []
                    for image in json.load(args.list)['images']:
                        filenames.append(image['file_name'])

            for filename in tqdm.tqdm(filenames):
                stem, ext = os.path.splitext(filename)
                if ext in ['.png', '.tiff', '.tif', '.jpg']:
                    filepath = root + '/' + filename
                    result = inference_detector(model, filepath)
                    filepath = filepath.replace(ext, '.json')
                    asap, contexts = create_asap(filepath)
                    if group not in asap['groups']:
                        asap['groups'].append(group)
                    group_id = asap['groups'].index(group)
                    contexts[group_id] = []
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
                    json.dump(asap, open(filepath, 'w'))


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
