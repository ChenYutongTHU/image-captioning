import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
import json
import cv2 

class CocoDataset(data.Dataset):
    def __init__(
        self, 
        image_ids_path, #/data/disk1/private/FXData/COCO/karpathy_image_ids/coco_train_image_id.txt
        input_seq,  #/data/disk1/private/FXData/COCO/sent/coco_train_input.pkl
        target_seq, #/data/disk1/private/FXData/COCO/sent/coco_train_target.pkl
        gv_feat_path, #''
        att_feats_folder,  #/data/disk1/private/FXData/COCO
        seq_per_img, 
        max_feat_num,
        id2name_path, #/data/disk1/private/FXData/COCO/id2name_123287.json
        annotation_path
    ):
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        self.image_ids = utils.load_lines(image_ids_path)  # a list of int
        self.att_feats_folder = att_feats_folder if len(att_feats_folder) > 0 else None
        self.gv_feat = pickle.load(open(gv_feat_path, 'rb'), encoding='bytes') if len(gv_feat_path) > 0 else None
        self.annotation_path = annotation_path
        with open(os.path.join(self.annotation_path,'bboxes_coco_123287.json'), 'r') as f:
            self.annotation_bboxes = json.load(f)
        self.id2category = utils.load_lines(os.path.join(self.annotation_path,'id2category.txt'))


        with open(id2name_path, 'r') as f:
            self.id2name = json.load(f)
        #None
        if input_seq is not None and target_seq is not None:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            self.seq_len = len(self.input_seq[self.image_ids[0]][0,:])# 
        else:
            self.seq_len = -1
            self.input_seq = None
            self.target_seq = None
         
    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def get_feature_path(self, image_id):
        name = self.id2name[image_id]
        split = name[5:8]
        if split == 'tra':
            split = split + 'in' #train
        feature_dir = os.path.join(self.att_feats_folder, '{}2014_output/features'.format(split))
        feature_path = os.path.join(feature_dir, '{}.npz'.format(name[:-4]))
        return feature_path

    def get_img_path(self, image_id):
        name = self.id2name[image_id]
        split = name[5:8]
        if split == 'tra':
            split = split + 'in' #train
        elif split == 'tes':
            split = split + 't'
        img_dir = os.path.join(self.att_feats_folder, '{}2014'.format(split))
        img_path = os.path.join(img_dir, name)
        return img_path   

    def get_coco_annotation(self, image_id):
        name = self.id2name[image_id]
        img_path = os.path.join(self.annotation_path, 'bboxes', name) 
        assert os.path.isfile(img_path), 'only annotations for Karpathy val/test are stored in disk'
        return img_path

    def get_coco_annotated_image(self, image_id):
        img_path = self.get_img_path(image_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = utils.draw_bbox(img, self.annotation_bboxes[image_id], self.id2category)
        return img


    def get_processedimg_path(self, image_id):
        name = self.id2name[image_id]
        split = name[5:8]
        if split == 'tra':
            split = split + 'in' #train
        elif split == 'tes':
            split = split + 't'
        img_dir = os.path.join(self.att_feats_folder, '{}2014_output/images'.format(split))
        img_path = os.path.join(img_dir, name)
        return img_path

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

        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int') #5,17
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
           
        n = len(self.input_seq[image_id])   
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)    #sample seq_per_img             
        else: #<=
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n) #complement
            input_seq[0:n, :] = self.input_seq[image_id]
            target_seq[0:n, :] = self.target_seq[image_id]
           
        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.input_seq[image_id][ix,:]
            target_seq[sid + i] = self.target_seq[image_id][ix,:]
        return indices, input_seq, target_seq, gv_feat, att_feats, image_id