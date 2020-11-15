import os
import torch
from torchvision import transforms
from lib.config import cfg
from datasets.coco_dataset import CocoDataset
import samplers.distributed
import numpy as np

def sample_collate(batch):
    #batch ((indice1, input_seq1,...),(indice2, input_seq2))
    #zip(*batch)
    #indices = (indice1, indice2)
    #input_seq = (input_seq1, input_seq2)
    #image_ids = (imageid1,id2,...)
    indices, input_seq, target_seq, gv_feat, att_feats, image_ids = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    image_ids = np.stack(image_ids, axis=0).reshape(-1)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)# b 5,L  5*bs,L
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    atts_num = [x.shape[0] for x in att_feats] #x = bbox, 2048
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)

    return indices, input_seq, target_seq, gv_feat, att_feats, att_mask, image_ids

def sample_collate_val(batch):
    indices, gv_feat, att_feats, image_ids = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    image_ids = np.stack(image_ids, axis=0).reshape(-1)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)

    return indices, gv_feat, att_feats, att_mask, image_ids


def load_train(distributed, epoch, coco_set):
    sampler = samplers.distributed.DistributedSampler(coco_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TRAIN.BATCH_SIZE, #10
        shuffle = shuffle,  #True
        num_workers = cfg.DATA_LOADER.NUM_WORKERS,  #4
        drop_last = cfg.DATA_LOADER.DROP_LAST,  #True
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, #True
        sampler = sampler, 
        collate_fn = sample_collate
    )
    return loader

def load_val(image_ids_path, gv_feat_path, att_feats_folder):
    coco_set = CocoDataset(
        image_ids_path = image_ids_path, 
        input_seq = None, 
        target_seq = None, 
        gv_feat_path = gv_feat_path, 
        att_feats_folder = att_feats_folder,
        seq_per_img = 1, 
        max_feat_num = cfg.DATA_LOADER.MAX_FEAT,
        id2name_path = cfg.DATA_LOADER.ID2NAME,
        annotation_path = cfg.DATA_LOADER.COCO_ANNOTATION
    )

    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TEST.BATCH_SIZE,
        shuffle = False, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = False, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, 
        collate_fn = sample_collate_val
    )
    return loader