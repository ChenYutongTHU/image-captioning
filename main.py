import os
import json
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np
import cv2
from tensorboardX import SummaryWriter
import jieba
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
from datasets import coco_dataset, aic_dataset, combined_dataset
import lib.utils as utils
from lib.utils import AverageMeter, ProgressBar
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            if args.local_rank == 0:
                self.writer = SummaryWriter(args.summary_dir)
                self.writer_val = {
                    'coco':SummaryWriter(os.path.join(args.summary_dir,'coco')),
                    'aic':SummaryWriter(os.path.join(args.summary_dir,'aic'))}
        else:
            self.writer = SummaryWriter(args.summary_dir)
            self.writer_val = {
                'coco':SummaryWriter(os.path.join(args.summary_dir,'coco')),
                'aic':SummaryWriter(os.path.join(args.summary_dir,'aic'))}

        self.device = torch.device("cuda")

        self.rl_stage = False
        self.setup_logging()
        self.setup_dataset()
        self.setup_network()
        if not self.distributed or dist.get_rank() == 0:
            self.val_evaler, self.test_evaler = {}, {}

            for dataset_name in self.dataset_dict:
                if dataset_name=='coco':
                    self.val_evaler[dataset_name] = Evaler(
                            eval_ids = cfg.COCO_DATA_LOADER.VAL_ID,
                            gv_feat = cfg.COCO_DATA_LOADER.VAL_GV_FEAT,
                            att_feats = cfg.COCO_DATA_LOADER.VAL_ATT_FEATS,
                            eval_annfile = cfg.INFERENCE.COCO_VAL_ANNFILE,
                            dataset_name = 'coco'
                        )
                    # self.test_evaler[dataset_name] = Evaler(
                    #     eval_ids = cfg.COCO_DATA_LOADER.TEST_ID,
                    #     gv_feat = cfg.COCO_DATA_LOADER.TEST_GV_FEAT,
                    #     att_feats = cfg.COCO_DATA_LOADER.TEST_ATT_FEATS,
                    #     eval_annfile = cfg.INFERENCE.COCO_TEST_ANNFILE,
                    #     dataset_name = 'coco'
                    # )
                elif dataset_name=='aic':
                    self.val_evaler[dataset_name] = Evaler(
                            eval_ids = cfg.AIC_DATA_LOADER.VAL_ID,
                            gv_feat = cfg.AIC_DATA_LOADER.VAL_GV_FEAT,
                            att_feats = cfg.AIC_DATA_LOADER.VAL_ATT_FEATS,
                            eval_annfile = cfg.INFERENCE.AIC_VAL_ANNFILE,
                            dataset_name = 'aic'
                        )
                    # self.test_evaler[dataset_name] = Evaler(
                    #     eval_ids = cfg.AIC_DATA_LOADER.TEST_ID,
                    #     gv_feat = cfg.AIC_DATA_LOADER.TEST_GV_FEAT,
                    #     att_feats = cfg.AIC_DATA_LOADER.TEST_ATT_FEATS,
                    #     eval_annfile = cfg.INFERENCE.AIC_TEST_ANNFILE,
                    #     dataset_name = 'aic'
                    # )             
            self.scorer = Scorer()

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return
        
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
        
        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device), 
                device_ids = [self.args.local_rank], 
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            self.model = torch.nn.DataParallel(model).cuda()

        if self.args.resume > 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                    map_location=lambda storage, loc: storage)
            )

        self.optim = Optimizer(self.model)
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()
        
    def setup_dataset(self):
        self.logger.info('Setting up dataset ...')
        self.dataset_dict = {}
        if 'coco' in self.args.dataset_name:
            self.coco_set = coco_dataset.CocoDataset(            
                image_ids_path = cfg.COCO_DATA_LOADER.TRAIN_ID,  #
                input_seq = cfg.COCO_DATA_LOADER.INPUT_SEQ_PATH,  #
                target_seq = cfg.COCO_DATA_LOADER.TARGET_SEQ_PATH, #
                gv_feat_path = cfg.COCO_DATA_LOADER.TRAIN_GV_FEAT,  #
                att_feats_folder = cfg.COCO_DATA_LOADER.TRAIN_ATT_FEATS, #
                seq_per_img = cfg.COCO_DATA_LOADER.SEQ_PER_IMG,#5
                max_feat_num = cfg.COCO_DATA_LOADER.MAX_FEAT,#-1
                id2name_path = cfg.COCO_DATA_LOADER.ID2NAME,
                annotation_path = cfg.COCO_DATA_LOADER.COCO_ANNOTATION
            )
            self.dataset_dict['coco'] = self.coco_set
        if 'aic' in self.args.dataset_name:
            self.aic_set = aic_dataset.AICDataset(            
                image_ids_path = cfg.AIC_DATA_LOADER.TRAIN_ID,  #
                input_seq = cfg.AIC_DATA_LOADER.INPUT_SEQ_PATH,  #
                target_seq = cfg.AIC_DATA_LOADER.TARGET_SEQ_PATH, #
                gv_feat_path = cfg.AIC_DATA_LOADER.TRAIN_GV_FEAT,  #
                att_feats_folder = cfg.AIC_DATA_LOADER.TRAIN_ATT_FEATS, #
                seq_per_img = cfg.AIC_DATA_LOADER.SEQ_PER_IMG,#5
                max_feat_num = cfg.AIC_DATA_LOADER.MAX_FEAT,#-1
                img_dir=cfg.AIC_DATA_LOADER.TRAIN_IMG_DIR,  #train/val
                processedimg_dir=cfg.AIC_DATA_LOADER.TRAIN_PROCESSEDIMG_DIR
            )
            self.dataset_dict['aic'] = self.aic_set
        self.combined_set = combined_dataset.CombinedDataset(
            datasets_dict = self.dataset_dict
        )
        self.logger.info('Finish setting up dataset ...')

    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.combined_set)

    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None
        val_results, test_results = {},{}
        for dataset_name in self.dataset_dict:
            val_results[dataset_name] = self.val_evaler[dataset_name](self.model, 'val_' + str(epoch))
            self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
            self.logger.info('######## {} ########'.format(dataset_name.upper()))
            self.logger.info(str(val_results[dataset_name]))
             
            # test_results[dataset_name] = self.test_evaler[dataset_name](self.model,'test_' + str(epoch + 1))
            # self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
            # self.logger.info('######## {} ########'.format(dataset_name.upper()))
            # self.logger.info(str(test_results[dataset_name]))

            val = 0
            for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
                val -= val_results[dataset_name][score_type] * weight
            for score_type in val_results[dataset_name]:
                self.writer_val[dataset_name].add_scalar(score_type, val_results[dataset_name][score_type], epoch+1)
            self.writer_val[dataset_name].add_scalar('weighted valuation', val, epoch)
#SCORER:
  # TYPES: ['CIDEr']
  # WEIGHTS: [1.0]
        return val

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model", epoch+1))

    def make_kwargs(self, indices, input_seq, target_seq, gv_feat, att_feats, att_mask):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask
        }
        return kwargs

    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.module.ss_prob = ss_prob


    def display(self, iteration, data_time, batch_time, losses, loss_info):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        #info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        #self.logger.info('Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
        # for name in sorted(loss_info):
        #     self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()

    def summary(self, iteration, loss, image_ids, dataset_name):
        if self.distributed and dist.get_rank() > 0:
            return
        if not iteration % self.args.summary_freq_scalar:
            self.writer.add_scalar('loss', loss.item(), iteration)
            self.writer.add_scalar('lr', self.optim.get_lr(), iteration)

        if not iteration % self.args.summary_freq_img2cap:
            id_ = image_ids[0].item()
            dataset_name = dataset_name[0]
            # annotated_image = self.coco_set.get_coco_annotated_image(id_)
            # self.writer.add_image('AnnotatedImage', img_tensor=annotated_image, global_step=iteration, dataformats='HWC')
            processed_img_filename = self.dataset_dict[dataset_name].get_processedimg_path(id_)
            #print(processed_img_filename)
            processed_img = cv2.imread(processed_img_filename)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)            
            self.writer.add_image('ProcessedImage_{}'.format(dataset_name), img_tensor=processed_img, global_step=iteration, dataformats='HWC')

    def forward(self, kwargs):
        if self.rl_stage == False:
            logit = self.model(**kwargs)
            loss, loss_info = self.xe_criterion(logit, kwargs[cfg.PARAM.TARGET_SENT])
        else:
            ids = kwargs[cfg.PARAM.INDICES]
            gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

            # max
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = True
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            self.model.eval()
            with torch.no_grad():
                seq_max, logP_max = self.model.module.decode(**kwargs)
            self.model.train()
            rewards_max, rewards_info_max = self.scorer(ids, seq_max.data.cpu().numpy().tolist())
            rewards_max = utils.expand_numpy(rewards_max)

            ids = utils.expand_numpy(ids)
            gv_feat = utils.expand_tensor(gv_feat, cfg.COCO_DATA_LOADER.SEQ_PER_IMG)
            att_feats = utils.expand_tensor(att_feats, cfg.COCO_DATA_LOADER.SEQ_PER_IMG)
            att_mask = utils.expand_tensor(att_mask, cfg.COCO_DATA_LOADER.SEQ_PER_IMG)

            # sample
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = False
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            seq_sample, logP_sample = self.model.module.decode(**kwargs)
            rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist())

            rewards = rewards_sample - rewards_max
            rewards = torch.from_numpy(rewards).float().cuda()
            loss = self.rl_criterion(seq_sample, logP_sample, rewards)
            
            loss_info = {}
            for key in rewards_info_sample:
                loss_info[key + '_sample'] = rewards_info_sample[key]
            for key in rewards_info_max:
                loss_info[key + '_max'] = rewards_info_max[key]

        return loss, loss_info

    def train(self):
        self.model.train()
        self.optim.zero_grad()

        iteration = 0
        start_epoch = 0
        if self.args.resume>0:
            start_epoch = self.args.resume
            self.setup_loader(start_epoch)
            iteration = start_epoch*len(self.training_loader)
            print('Start from epoch {} iteration {}'.format(start_epoch, iteration))
            for i in range(start_epoch):
                for j in range(iteration):
                    self.optim.scheduler_step('Iter')
                self.optim.scheduler_step('Epoch', None)
            print('Learning rate {}'.format(self.optim.get_lr()))
            print('Press enter to continue')
            input()
            print('')

        for epoch in  range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            if epoch == cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            self.setup_loader(epoch)

            start = time.time()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            losses = AverageMeter()
            if not self.distributed or self.args.local_rank == 0:
                pbar = ProgressBar(n_total=len(self.training_loader), desc='Training')
                val = self.eval(epoch)
                print()
            for step, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask, image_ids, dataset_name) in enumerate(self.training_loader):

                data_time.update(time.time() - start)

                input_seq = input_seq.cuda()
                target_seq = target_seq.cuda()
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()

                kwargs = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)
                loss, loss_info = self.forward(kwargs)
                loss.backward()

                if step%cfg.DATA_LOADER.ACCUMULATE_STEPS == 0:                    
                    utils.clip_gradient(self.optim.optimizer, self.model,
                        cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                    self.optim.step()
                    self.optim.zero_grad()
                    self.optim.scheduler_step('Iter')
                
                batch_time.update(time.time() - start)
                start = time.time()
                losses.update(loss.item())

                self.summary(iteration, loss, image_ids, dataset_name)
                self.display(iteration, data_time, batch_time, losses, loss_info)
                iteration += 1

                if self.distributed:
                    dist.barrier()
                if not self.distributed or self.args.local_rank == 0:
                    pbar(step)
            
            print()
            self.save_model(epoch)
           # val = self.eval(epoch)

            self.optim.scheduler_step('Epoch', None)
            self.scheduled_sampling(epoch)

            if self.distributed:
                dist.barrier()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', type=str, default='debug')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument('--dataset_name', default='coco_aic')
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--config', default='config.yml')
    #summary
    parser.add_argument('--summary_freq_scalar', type=int, help='per iter')
    parser.add_argument('--summary_freq_img2cap', type=int, help='per iter')

    parser.add_argument('--accumulate_steps', type=int, help='a * n_gpu * bs_forward = 40')
    parser.add_argument('--batchsize_forward', type=int, help='a * n_gpu * bs_forward = 40')


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    cfg.DATA_LOADER.ACCUMULATE_STEPS = args.accumulate_steps
    cfg.TRAIN.BATCH_SIZE = args.batchsize_forward
    args.summary_dir = os.path.join(args.folder,'train_summary')
    #print(args.summary_dir)
    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)
    if not os.path.exists(os.path.join(args.summary_dir,'coco')):
        os.makedirs(os.path.join(args.summary_dir,'coco'))
    if not os.path.exists(os.path.join(args.summary_dir, 'aic')):
        os.makedirs(os.path.join(args.summary_dir,'aic'))
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(args.config)
    cfg.ROOT_DIR = args.folder
    trainer = Trainer(args)
    trainer.train()
