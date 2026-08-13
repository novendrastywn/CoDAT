# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from datasets.mixup import Mixup
from timm.models import create_model


import utils
import contextlib
from models import *

from analysis import get_model_complexity_info


def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, 
                        metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=4)
    parser.add_argument('--test_num_crop', type=int, default=3)

    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--input_shape', type=str, default='NCHW') # NCTHW or NCHW
    parser.add_argument('--num_frames', type=int, default=8)

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
 

    known_args, _ = parser.parse_known_args()

    return parser.parse_args()


def main():

    args = get_args()

    
    if args.input_shape == "NCTHW":
        input_shape = tuple(( 3, args.num_frames, args.input_size, args.input_size))
        input_batches = True
    else:
        input_shape = tuple((args.num_frames, 3, args.input_size, args.input_size))
        input_batches = False

    # else:
    #     raise ValueError('invalid input shape')

    # input_tensor = torch.randn(input_shape)

    if 'deit' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            fc_drop_rate=args.fc_drop_rate,
            drop_path_rate=args.drop_path,
            # kernel_size=args.tubelet_size,
            num_frames=args.num_frames,
        )
    elif 'par_action_s1' in args.model or 'par_action_s2' in args.model or 'par_action_s3' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            num_segments=args.num_frames * args.num_segments,
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,

        )
    model.eval()
    # model.backbone = model.backbone.switch_to_deploy()
    # print(model)

    analysis_results = get_model_complexity_info(model, input_shape, input_batches=input_batches)
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    table = analysis_results['out_table']
    print(table)
    split_line = '=' * 30
    print(f'\n{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

    # print(f"{args.model}, params: {params}, macs: {macs} at input size {input_shape}")


if __name__ == '__main__':
    main()
