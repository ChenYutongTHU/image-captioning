import os
import torch
from torchvision import transforms
from lib.config import cfg
from datasets import coco_dataset, aic_dataset, combined_dataset
import samplers.distributed
import numpy as np
def padding_sentences(sentences, max_length=None, padding_index=0):
    #target -1  
    #input 0
    if type(sentences[0][0])==list: # [[5*s],[5*s],...,[5*s]] or [s11,s12,s13,...]
        sentences_ = []
        for ss in sentences:#[s1,s2,s3,...,s4,s5]
            sentences_ += ss 
        sentences = sentences_
    if max_length==None:
        max_length = max([len(s) for s in sentences])
    pad_sentences = []
    for s in sentences:
        pad_s = s+[padding_index]*(max_length-len(s))
        pad_sentences.append(pad_s)
    return pad_sentences, max_length

def sample_collate(batch):
    #batch [(indice1, input_seq1,...),(indice2, input_seq2)]
    #zip(*batch)
    #indices = (indice1, indice2)
    #input_seq = (input_seq1, input_seq2)
    #image_ids = (imageid1,id2,...)
    dataset_names  = [b[1] for b in batch]
    info_items = [b[0] for b in batch]
    indices, input_seq, target_seq, gv_feat, att_feats, image_ids = zip(*info_items)


    indices = np.stack(indices, axis=0).reshape(-1)
    image_ids = np.stack(image_ids, axis=0).reshape(-1)
    # input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)# b 5,L  5*bs,L
    # target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    input_seq, max_input_length = padding_sentences(input_seq, max_length=None, padding_index=0)
    target_seq, _ = padding_sentences(target_seq, max_length=max_input_length, padding_index=-1)
    input_seq = torch.LongTensor(input_seq)
    target_seq = torch.LongTensor(target_seq)

    #input_lengths = [b.shape[1] for b in input_seq] #5,L
    #max_input_length = max(input_lengths)

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

    return indices, input_seq, target_seq, gv_feat, att_feats, att_mask, image_ids, dataset_names

def sample_collate_val(batch):
    indices,  gv_feat, att_feats, image_ids = zip(*batch)
    
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


def load_train(distributed, epoch, data_set):
    sampler = samplers.distributed.DistributedSampler(data_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    loader = torch.utils.data.DataLoader(
        data_set, 
        batch_size = cfg.TRAIN.BATCH_SIZE, #10
        shuffle = shuffle,  #True
        num_workers = cfg.DATA_LOADER.NUM_WORKERS,  #4
        drop_last = cfg.DATA_LOADER.DROP_LAST,  #True
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, #True
        sampler = sampler, 
        collate_fn = sample_collate
    )
    return loader

def load_val(image_ids_path, gv_feat_path, att_feats_folder, dataset_name):
    if dataset_name=='coco':
        dataset = coco_dataset.CocoDataset(
            image_ids_path = image_ids_path, 
            input_seq = None, 
            target_seq = None, 
            gv_feat_path = gv_feat_path, 
            att_feats_folder = att_feats_folder,
            seq_per_img = 1, 
            max_feat_num = cfg.COCO_DATA_LOADER.MAX_FEAT,
            id2name_path = cfg.COCO_DATA_LOADER.ID2NAME,
            annotation_path = cfg.COCO_DATA_LOADER.COCO_ANNOTATION
        )
    elif dataset_name=='aic': #
        dataset = aic_dataset.AICDataset(
            image_ids_path = image_ids_path,  #
            input_seq = None,  #
            target_seq = None, #
            gv_feat_path = gv_feat_path,  #
            att_feats_folder = att_feats_folder, #
            seq_per_img = 1,#5
            max_feat_num = cfg.AIC_DATA_LOADER.MAX_FEAT,#-1
            img_dir=cfg.AIC_DATA_LOADER.VAL_IMG_DIR,  #val/test the same
            processedimg_dir=cfg.AIC_DATA_LOADER.VAL_PROCESSEDIMG_DIR  #val/test the same
        )

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = cfg.TEST.BATCH_SIZE,
        shuffle = False, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = False, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, 
        collate_fn = sample_collate_val
    )
    return loader