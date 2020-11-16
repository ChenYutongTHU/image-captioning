import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
import json
import cv2 

class BasicDataset(data.Dataset):
    def __init__(
        self, 
        image_ids_path, #
        input_seq,  #
        target_seq, #
        gv_feat_path, #''
        att_feats_folder,  #
        seq_per_img, 
        max_feat_num
    ):
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        self.image_ids = utils.load_lines(image_ids_path)  # a list of str
        self.att_feats_folder = att_feats_folder if len(att_feats_folder) > 0 else None
        self.gv_feat = pickle.load(open(gv_feat_path, 'rb'), encoding='bytes') if len(gv_feat_path) > 0 else None
        if input_seq is not None and target_seq is not None:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            self.seq_len = 0#len(self.input_seq[self.image_ids[0]][0,:])
        else:
            self.seq_len = -1
            self.input_seq = None
            self.target_seq = None
        return

    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def get_feature_path(self, image_id):
        pass

    def get_img_path(self, image_id):
        pass

    def get_processedimg_path(self, image_id):
        pass

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        indices = np.array([index]).astype('int')

        if self.gv_feat is not None:
            gv_feat = self.gv_feat[image_id]
            gv_feat = np.array(gv_feat).astype('float32')
        else:
            gv_feat = np.zeros((1,1))

        if self.att_feats_folder is not None:
            att_feats = np.load(self.get_feature_path(image_id))['feat']
            att_feats = np.array(att_feats).astype('float32')
        else:
            att_feats = np.zeros((1,1))
        
        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:# -1
           att_feats = att_feats[:self.max_feat_num, :]

        if self.seq_len < 0:
            return indices, gv_feat, att_feats, image_id

        input_seq = []#np.zeros((self.seq_per_img, self.seq_len), dtype='int') #5,17
        target_seq = []#np.zeros((self.seq_per_img, self.seq_len), dtype='int')
           
        n = len(self.input_seq[image_id])   
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)    #sample seq_per_img             
        else: #<=
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n) #complement
            input_seq = self.input_seq[image_id]
            target_seq = self.target_seq[image_id]
           
        for i, ix in enumerate(ixs):
            input_seq.append(self.input_seq[image_id][ix])
            target_seq.append(self.target_seq[image_id][ix])

        return indices, input_seq, target_seq, gv_feat, att_feats, image_id