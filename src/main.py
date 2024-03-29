import yaml
import sys
import argparse
import os
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR
import warnings
warnings.filterwarnings("ignore")
import torch
torch.backends.cudnn.enabled = False
import numpy as np
import random

def main(config, args):
    Mission = TextSR(config, args)

    if args.test:
        Mission.test()
    elif args.demo:
        Mission.demo()
    else:
        Mission.train()

def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tsrn', choices=['tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
                                                           'edsr', 'laprn'])
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default=None, help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--sync', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--sync_real', action='store_true', default=False, help='use bsr synthetic LR')
    parser.add_argument('--sync_all', action='store_true', default=False, help='use bsr synthetic LR only')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--content', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--conv_num', type=int, default=2)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--nonlocal_type', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='../result')
    parser.add_argument('--config_path', '-c', type=str, default="./config")
    args = parser.parse_args()
    config_path = os.path.join(args.config_path, 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    seed_torch(args.seed)
    main(config, args)
