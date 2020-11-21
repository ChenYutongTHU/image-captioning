import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

import sys
sys.path.append(cfg.INFERENCE.COCO_PATH)
print(cfg.INFERENCE.COCO_PATH)
from my_pycocotools.coco import COCO
from my_pycocoevalcap.eval import COCOEvalCap


class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = torch.device("cuda")

        self.setup_logging()
        self.setup_network()
        if self.args.test_raw_image:
            self.raw_evaler = Evaler(
                        eval_ids = cfg.RAW_DATA_LOADER.TEST_IMG_DIR,
                        gv_feat = None, 
                        att_feats = cfg.RAW_DATA_LOADER.TEST_ATT_FEATS,
                        dataset_name = 'raw',
                        eval_annfile = None
                    )
            self.evaler = {'raw': self.raw_evaler}
        else:
            self.coco_evaler = Evaler(
                        eval_ids = cfg.COCO_DATA_LOADER.TEST_ID,
                        gv_feat = cfg.COCO_DATA_LOADER.TEST_GV_FEAT,
                        att_feats = cfg.COCO_DATA_LOADER.TEST_ATT_FEATS,
                        eval_annfile = cfg.INFERENCE.COCO_TEST_ANNFILE,
                        dataset_name = 'coco'
                    )
            self.aic_evaler = Evaler(
                        eval_ids = cfg.AIC_DATA_LOADER.TEST_ID,
                        gv_feat = cfg.AIC_DATA_LOADER.TEST_GV_FEAT,
                        att_feats = cfg.AIC_DATA_LOADER.TEST_ATT_FEATS,
                        eval_annfile = cfg.INFERENCE.AIC_TEST_ANNFILE,
                        dataset_name = 'aic'
                    )     
            self.evaler = {'aic': self.aic_evaler,'coco': self.coco_evaler}        

    def setup_logging(self):
        cfg.LOGGER_NAME = 'test_{}_log'.format(self.args.resume)
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)
        self.model = torch.nn.DataParallel(model).cuda()
        if self.args.resume > 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                    map_location=lambda storage, loc: storage)
            )
        
    def eval(self, epoch):
        for dataset_name in self.evaler:
            res = self.evaler[dataset_name](self.model, 'test_' + str(epoch), output_attention=cfg.INFERENCE.OUTPUT_ATTENTION)
            self.logger.info('########{} Epoch '.format(dataset_name) + str(epoch) + ' ########'.format(dataset_name))
            self.logger.info(str(res))

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
    parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--test_raw_image', action='store_true', default=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        if not os.path.exists(args.config):
            config_path = os.path.join(args.folder, args.config)
        else:
            config_path = args.config
        cfg_from_file(config_path)
    cfg.ROOT_DIR = args.folder

    tester = Tester(args)
    tester.eval(args.resume)
