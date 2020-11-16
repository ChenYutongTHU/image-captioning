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

class Evaler(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        att_feats,
        eval_annfile,
        dataset_name
    ):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        self.eval_ids = np.array(utils.load_lines(eval_ids))#np.array(utils.load_ids(eval_ids))
        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats, dataset_name)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile) 
        self.dataset_name = dataset_name

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs
        
    def __call__(self, model, rname):
        model.eval()
        
        results = []
        cnt = 0
        with torch.no_grad():
            for _, (indices, gv_feat, att_feats, att_mask, image_ids) in tqdm.tqdm(enumerate(self.eval_loader)):
                ids = self.eval_ids[indices] #=image_id #str
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _ = model.module.decode_beam(**kwargs)
                else:
                    seq, _ = model.module.decode(**kwargs)
                sents = utils.decode_sequence(self.vocab, seq.data)
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
        eval_res = self.evaler.eval(results) #...

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))


        model.train()
        return eval_res