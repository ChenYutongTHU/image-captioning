import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg
import cv2

class Evaler(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        att_feats,
        dataset_name,
        eval_annfile=None,
    ):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)
        if dataset_name == 'raw':
            self.eval_ids = np.array([img.split('.')[0] for img in os.listdir(eval_ids)])
            self.evaler = None
        else:
            self.eval_ids = np.array(utils.load_lines(eval_ids))#np.array(utils.load_ids(eval_ids))
            self.evaler = evaluation.create(dataset_name, eval_annfile) 
        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats, dataset_name)
        self.dataset_name = dataset_name

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask, output_attention=False):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        kwargs['output_attention'] = output_attention
        return kwargs
    
    def visualize_attention(self, sents, attention_scores, att_mask, image_ids, number=10, niter=0):
        result_folder = os.path.join(cfg.ROOT_DIR, 'result','attention_visualization')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        batch_size, head_size = attention_scores.shape[0], attention_scores.shape[1]
        for i in range(batch_size):
            if i>number:
                break
            mask = att_mask[i] # N (1-> valid 0->padding)
            score = attention_scores[i] #H,N,T
            tokens = sents[i].split(' ') #
            img0 = cv2.imread(self.eval_loader.dataset.get_img_path(image_ids[i]))
            vg_instance = self.eval_loader.dataset.get_vginstance(image_ids[i])

            for h in range(head_size):
                score_h = score[h] # N,T
                sub_folder = os.path.join(result_folder, image_ids[i], 'head'+str(h))
                if not os.path.exists(sub_folder):
                    os.makedirs(sub_folder)  
                for ti, t in enumerate(tokens):
                    img = img0.copy()
                    bg = np.ones_like(img0, dtype=np.float32)*0.1
                    H, W = bg.shape[:2]
                    for bb, sc in zip(vg_instance['bounding_boxes'], score_h[:,ti].detach().cpu().numpy()):
                        x1,y1,x2,y2 = bb.astype(np.int32)
                        [x1,x2] = np.clip([x1,x2],0,W)
                        [y1,y2] = np.clip([y1,y2],0,H)
                        bg[y1:y2,x1:x2] += sc
                    img = (np.clip(img*bg,0,255)).astype(np.int32)
                    box_id = torch.argmax(score_h[:, ti]).item()
                    box = vg_instance['bounding_boxes'][box_id]
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255,0,0], 2)
                    cv2.rectangle(img, (int(box[0]+5), int(box[1]+5)), (int(box[0]+12*len(t)), int(box[1]+20)), [255,255,255], -1)
                    cv2.putText(img, text=t, org=(int(box[0]+5), int(box[1]+15)), fontFace=3, fontScale=0.5, thickness=1, color=[0,0,0])                    
                    cv2.imwrite(os.path.join(sub_folder,'{}_{}.jpg'.format(ti,t)), img)

                #input()


    def __call__(self, model, rname, output_attention=False):
        model.eval()
        
        results = []
        cnt = 0
        with torch.no_grad():
            for _, (indices, gv_feat, att_feats, att_mask, image_ids) in tqdm.tqdm(enumerate(self.eval_loader)):
                ids = self.eval_ids[indices] #=image_id #str
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask, output_attention)
                if kwargs['BEAM_SIZE'] > 1:
                    if kwargs['output_attention']:
                        seq, _, attention_scores, att_mask = model.module.decode_beam(**kwargs)
                    else:
                        seq, _ = model.module.decode_beam(**kwargs)
                else:
                    seq, _ = model.module.decode(**kwargs) 
                sents = utils.decode_sequence(self.vocab, seq.data)
                if cnt==0 and output_attention:
                    self.visualize_attention(sents, attention_scores, att_mask, image_ids, niter=0)
                for sid, sent in enumerate(sents):
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]) if self.dataset_name=='coco' else str(ids[sid]), 
                        cfg.INFERENCE.CAP_KEY: sent}
                    #ids[sid] image_id
                    #sent word untokenized
                    results.append(result)
                # if cnt>=3:
                #     break
                cnt += 1
                #break #!!
        if self.evaler:
            eval_res = self.evaler.eval(results)
        else:
            eval_res = None
        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, self.dataset_name+'+result_' + rname +'.json'), 'w'))


        model.train()
        return eval_res