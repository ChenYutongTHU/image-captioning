import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
import json
import cv2 
from datasets.basic_dataset import BasicDataset

import sys
from lib.config import cfg
sys.path.append(cfg.INFERENCE.COCO_PATH)
print(cfg.INFERENCE.COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


class CocoDataset(BasicDataset):
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
        super().__init__(image_ids_path, input_seq, target_seq, 
            gv_feat_path, att_feats_folder, seq_per_img, max_feat_num)
        self.annotation_path = annotation_path
        with open(os.path.join(self.annotation_path,'bboxes_coco_123287.json'), 'r') as f:
            self.annotation_bboxes = json.load(f)
        self.id2category = utils.load_lines(os.path.join(self.annotation_path,'id2category.txt'))
        with open(id2name_path, 'r') as f:
            self.id2name = json.load(f)

         
    def get_vginstance(self, image_id):
        name = self.id2name[image_id]
        split = name[5:8]
        if split == 'tra':
            split = split + 'in' #train
        feature_dir = os.path.join(self.att_feats_folder, '{}2014_output/information'.format(split))
        feature_path = os.path.join(feature_dir, '{}.pkl'.format(name[:-4]))
        import sys
        sys.path.append(cfg.INFERENCE.COCO_PATH)
        #print(cfg.INFERENCE.COCO_PATH)
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
        instance = pickle.load(open(feature_path, 'rb'), encoding='bytes')
        return instance

    def get_feature_path(self, image_id):
        name = self.id2name[image_id]
        split = name[5:8]
        if split == 'tra':
            split = split + 'in' #train
        feature_dir = os.path.join(self.att_feats_folder, '{}2014_output/features'.format(split))
        feature_path = os.path.join(feature_dir, '{}.npz'.format(name[:-4]))
        # feature_dir =self.att_feats_folder #os.path.join(self.att_feats_folder, '{}2014_output/features'.format(split))
        # feature_path = os.path.join(feature_dir, '{}.npz'.format(image_id))
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


    def get_annotatedimg_path(self, image_id):
        name = self.id2name[image_id]
        img_path = os.path.join(self.annotation_path, 'bboxes', name) 
        assert os.path.isfile(img_path), 'only annotations for Karpathy val/test are stored in disk'
        return img_path
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

    def get_coco_annotated_image(self, image_id):
        img_path = self.get_img_path(image_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = utils.draw_bbox(img, self.annotation_bboxes[image_id], self.id2category)
        return img



    def __len__(self):
        return len(self.image_ids)
