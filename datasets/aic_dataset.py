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

class AICDataset(BasicDataset):
    def __init__(
        self, 
        image_ids_path, #ai_challenger_caption_train_20170902_outputs/features
        input_seq,  #
        target_seq, #
        gv_feat_path, #''
        att_feats_folder,  #
        seq_per_img, 
        max_feat_num,
        img_dir,  #train/val
        processedimg_dir 
    ):
        super().__init__(image_ids_path, input_seq, target_seq, 
            gv_feat_path, att_feats_folder, seq_per_img, max_feat_num)
        self.img_dir = img_dir
        self.processedimg_dir = processedimg_dir

    def get_feature_path(self, image_id):
        feature_path = os.path.join(self.att_feats_folder, '{}.npz'.format(image_id))
        return feature_path

    def get_vginstance(self, image_id):
        feature_dir = os.path.join(self.att_feats_folder, '../information_2')
        feature_path = os.path.join(feature_dir, '{}.pkl'.format(image_id))
        instance = pickle.load(open(feature_path, 'rb'))
        return instance


    def get_img_path(self, image_id):
        img_path = os.path.join(self.img_dir, '{}.jpg'.format(image_id))
        return img_path

    def get_processedimg_path(self, image_id):
        img_path = os.path.join(self.processedimg_dir, '{}.jpg'.format(image_id))
        return img_path