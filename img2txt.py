import os, json, sys, random, pprint
import time, logging, argparse

import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter, ProgressBar
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file


CONFIG_PATH = 'caption_utils/config_zh.yml'
MODEL_PATH = 'caption_utils/caption_model_15.pth'
VOCAB_PATH = 'caption_utils/zh_vocabulary_bpe.txt'
class Img2Txt():
    def __init__(self, device):
        cfg_from_file(CONFIG_PATH)
        random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed_all(cfg.SEED)

        self.device = torch.device(device)
        self.setup_logging()
        self.load_model(model_path=MODEL_PATH)
        self.misc()


    def load_model(self, model_path):
        self.logger.info('Loading model {}'.format(model_path))
        self.model = models.create(cfg.MODEL.TYPE)
        self.model = torch.nn.DataParallel(self.model, device_ids=[self.device])
        assert os.path.isfile(model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def misc(self):
        self.logger.info('Loading vocabulary {}'.format(VOCAB_PATH))
        self.vocab = utils.load_vocab(VOCAB_PATH)
        self.max_feat_num = -1
        self.seq_len = -1

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)   
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        start_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME+start_time+'.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info(start_time)
        self.logger.info(pprint.pformat(cfg))

    def generate_caption(self, region_features): 
        start_time = time.time()
        #region features N*2014
        attn_feats = np.array(region_features).astype('float32')
        attn_feats = np.expand_dims(attn_feats, axis=0) #batch_size = 1
        attn_feats = torch.tensor(attn_feats, device=self.device)

        gv_feat = torch.zeros([1,1], device=self.device)
        att_mask = torch.ones([1, attn_feats.shape[-2]], device=self.device)

        kwargs = {}
        #kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = attn_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE  #3
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE #True
        kwargs['device'] = self.device
        kwargs['output_attention'] = False

        if kwargs['BEAM_SIZE'] >= 1:
            seq, _, _ = self.model.module.decode_beam(**kwargs)
        else:
            seq, _ = self.model.module.decode(**kwargs) 
        sents = utils.decode_sequence(self.vocab, seq.data, lang='zh')
        self.logger.info('Generate Caption Time cost {:.2f}'.format(time.time()-start_time))
        self.logger.info(sents)
        #print(sents)
        return sents

if __name__ == '__main__':
    example = Img2Txt(device='cuda:2')
    test_dir = '/data/disk1/private/chenyutong/Img2Poet/TEST/animals/vg/features/'
    for i in range(10):
        region_features = np.load(os.path.join(test_dir,str(i+1)+'.npz'))['feat']
        example.generate_caption(region_features)
    