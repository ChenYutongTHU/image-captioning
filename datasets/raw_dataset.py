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

class RawDataset(BasicDataset):
    def __init__(
        self, 
        img_dir, 
        att_feats_folder,  #
        max_feat_num,
        processedimg_dir 
    ):
        
        self.img_dir = img_dir
        self.image_ids = [img.split('.')[0] for img in os.listdir(self.img_dir)]
        self.att_feats_folder = att_feats_folder
        self.gv_feat = None
        self.max_feat_num = max_feat_num
        self.processedimg_dir = processedimg_dir
        self.seq_len = -1
        self.input_seq = None
        self.target_seq = None
        return 

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