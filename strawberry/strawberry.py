# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from pathlib import Path
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from .lib.config import cfg
from .lib.config import update_config
from .lib.core.loss import JointsMSELoss
from .lib.core.function import validate
from .lib.utils.utils import create_logger
from .tools.data_processing import create_coco_json, init_and_rename_data
from .tools.keypoint_visualization import kpt_visualization

import strawberry.lib.dataset
import strawberry.lib.models


def _parse_args(predefined=""):
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args(predefined.split())

    return args


def run_strawberry(temp_dir: Path):
    init_and_rename_data(
        temp_dir/'image', Path('strawberry/data/validation/image'))
    init_and_rename_data(
        temp_dir/'ply', Path('strawberry/data/validation/ply'))
    # create_coco_json()

    # origin code
    args = _parse_args(
        "--cfg strawberry/experiments/keypoint_strawberry.yaml TEST.MODEL_FILE strawberry/models/final_state_hj.pth TEST.USE_GT_BBOX True")
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * torch.cuda.device_count()
        logger.info("Let's use %d GPUs!" % torch.cuda.device_count())

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('strawberry.lib.models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=True)  # False
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        cfg=cfg,
        target_type=cfg.MODEL.TARGET_TYPE,
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('strawberry.lib.dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        # batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    logger.info('=> Start testing...')

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             str(temp_dir),  # final_output_dir,
             str(temp_dir)  # tb_log_dir)
             )

    ######
    return kpt_visualization(temp_dir)


if __name__ == '__main__':
    run_strawberry()
