import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import blocks
import lib.utils as utils
from lib.config import cfg
from models.basic_model import BasicModel

class AttBasicModel(BasicModel):
    def __init__(self):
        super(AttBasicModel, self).__init__()
        self.ss_prob = 0.0                               # Schedule sampling probability
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1       # include <BOS>/<EOS>
        self.att_dim = cfg.MODEL.ATT_FEATS_EMBED_DIM \
            if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0 else cfg.MODEL.ATT_FEATS_DIM

        # word embed
        sequential = [nn.Embedding(self.vocab_size, cfg.MODEL.WORD_EMBED_DIM)]
        sequential.append(utils.activation(cfg.MODEL.WORD_EMBED_ACT))
        if cfg.MODEL.WORD_EMBED_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.WORD_EMBED_DIM))
        if cfg.MODEL.DROPOUT_WORD_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED))
        self.word_embed = nn.Sequential(*sequential)

        # global visual feat embed
        sequential = []
        if cfg.MODEL.GVFEAT_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.GVFEAT_DIM, cfg.MODEL.GVFEAT_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.GVFEAT_EMBED_ACT))
        if cfg.MODEL.DROPOUT_GV_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_GV_EMBED))
        self.gv_feat_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        # attention feats embed
        sequential = []
        if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
        if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
        if cfg.MODEL.ATT_FEATS_NORM == True:
            sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
        self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        self.dropout_lm  = nn.Dropout(cfg.MODEL.DROPOUT_LM) if cfg.MODEL.DROPOUT_LM > 0 else None
        self.logit = nn.Linear(cfg.MODEL.RNN_SIZE, self.vocab_size)
        self.p_att_feats = nn.Linear(self.att_dim, cfg.MODEL.ATT_HIDDEN_SIZE) \
            if cfg.MODEL.ATT_HIDDEN_SIZE > 0 else None

        # bilinear
        if cfg.MODEL.BILINEAR.DIM > 0:
            self.p_att_feats = None
            self.encoder_layers = blocks.create(
                cfg.MODEL.BILINEAR.ENCODE_BLOCK, 
                embed_dim = cfg.MODEL.BILINEAR.DIM, 
                att_type = cfg.MODEL.BILINEAR.ATTTYPE,
                att_heads = cfg.MODEL.BILINEAR.HEAD,
                att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM,
                att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT,
                dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
                layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
            )

    def init_hidden(self, batch_size):
        return [Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda())]

    def make_kwargs(self, wt, gv_feat, att_feats, att_mask, p_att_feats, state, **kgs):
        kwargs = kgs
        kwargs[cfg.PARAM.WT] = wt
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs[cfg.PARAM.STATE] = state
        return kwargs

    def preprocess(self, **kwargs):
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        # embed gv_feat
        if self.gv_feat_embed is not None:
            gv_feat = self.gv_feat_embed(gv_feat)
        
        # embed att_feats
        if self.att_embed is not None:    
            att_feats = self.att_embed(att_feats)

        p_att_feats = self.p_att_feats(att_feats) if self.p_att_feats is not None else None

        # bilinear
        if cfg.MODEL.BILINEAR.DIM > 0:
            gv_feat, att_feats = self.encoder_layers(gv_feat, att_feats, att_mask) #encode block
            keys, value2s = self.attention.precompute(att_feats, att_feats)
            p_att_feats = torch.cat([keys, value2s], dim=-1)

        return gv_feat, att_feats, att_mask, p_att_feats

    # gv_feat -- batch_size * cfg.MODEL.GVFEAT_DIM
    # att_feats -- batch_size * att_num * att_feats_dim
    def forward(self, **kwargs): 
        seq = kwargs[cfg.PARAM.INPUT_SENT] 
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        p_att_feats = utils.expand_tensor(p_att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)

        outputs = Variable(torch.zeros(batch_size, seq.size(1), self.vocab_size).cuda())
        for t in range(seq.size(1)):
            if self.training and t >=1 and self.ss_prob > 0:
                prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                mask = prob < self.ss_prob
                if mask.sum() == 0:
                    wt = seq[:,t].clone()
                else:
                    ind = mask.nonzero().view(-1)
                    wt = seq[:, t].data.clone()
                    prob_prev = torch.exp(outputs[:, t-1].detach())
                    wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
            else:
                wt = seq[:,t].clone()

            if t >= 1 and seq[:, t].max() == 0:
                break
            
            kwargs = self.make_kwargs(wt, gv_feat, att_feats, att_mask, p_att_feats, state)
            output, state = self.Forward(**kwargs)
            if self.dropout_lm is not None:
                output = self.dropout_lm(output)

            logit = self.logit(output)
            outputs[:, t] = logit

        return outputs

    def get_logprobs_state(self, **kwargs):
        Forward_output = self.Forward(**kwargs)
        logprobs = F.log_softmax(self.logit(Forward_output[0]), dim=1) #B, V
        state = Forward_output[1]
        #print('get_logprobs_state\n',type(Forward_output[2]), len(Forward_output[2]),len(Forward_output[2][0]))
        if 'output_attention' in kwargs and kwargs['output_attention']:
            return [logprobs, state, Forward_output[2]]
        else:
            return [logprobs, state]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, state, selected_beam):
        shape = [int(sh) for sh in state.shape]#[#Layer, B, d]
        beam = selected_beam  #B, beam_size
        for _ in shape[2:]:
            beam = beam.unsqueeze(-1)   #B, beam_size, 1
        #beam B, beam_size, 1
        beam = beam.unsqueeze(0) #1, B, beam_size, 1
        
        state = torch.gather(
            state.view(*([shape[0], batch_size, cur_beam_size] + shape[2:])), 2, ##Layer, B, cur_beam_size, d
            beam.expand(*([shape[0], batch_size, beam_size] + shape[2:])) ##Layer, B, beam_size, d
        )
        state = state.view(*([shape[0], -1, ] + shape[2:]))
        return state

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        #print('decode beam!')
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        att_mask0 = att_mask
        beam_size = kwargs['BEAM_SIZE']
        output_attention = kwargs['output_attention']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs, attention_scores = [], []
        distributions = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = self.init_hidden(batch_size)
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())

        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size #???

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_state_output = self.get_logprobs_state(**kwargs)# word_logprob B, #V
            #print(len(logprobs_state_output))
            #print(logprobs_state_output[0].shape, logprobs_state_output[1].shape)
            word_logprob, state = logprobs_state_output[0], logprobs_state_output[1]
            if output_attention:
                attention_score = logprobs_state_output[2][-1] #[(36,8,67)] the last layer
                attention_score = attention_score.view(batch_size, -1, 
                    attention_score.shape[1], attention_score.shape[2])
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob # B,1,1 + B,1,#V

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                #B, cur_beam_size !=0  unended True ended False    -> B, cur_beam_size, 1
                seq_mask = seq_mask * mask #B, beam_size, 1  update seq_mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob) 
                #word_logprob B, beam_size #V 
                #seq_mask B, beam_size, 1
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                #seq_logprob B,1,1 -> B, beam_size, #V
                old_seq_logprob[:, :, 1:] = -999   #B, beam_size, 
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx / candidate_logprob.shape[-1] #B, bs
            #print('selected_beam', selected_beam.shape, selected_beam[0])
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
            #print('selected_words', selected_words.shape, selected_words[0]) #B, bs
            #input()
            for s in range(len(state)):
                #print('state shape before expand {}'.format(state[s].shape))
                state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)
                # print('state shape after expand {}'.format(state[s].shape))
                # input()
            #for a in range(len(attention_score)):
            if output_attention:
                selected_beam_ex = selected_beam.unsqueeze(-1).unsqueeze(-1).expand(batch_size, beam_size, attention_score.shape[-2], attention_score.shape[-1])
                this_word_attention_score = torch.gather(attention_score, 1, selected_beam_ex)
                #print('this word attention score shape {}'.format(this_word_attention_score.shape))
                attention_scores = list(
                    torch.gather(a, 1, selected_beam_ex) for a in attention_scores)
                attention_scores.append(this_word_attention_score)#B, bs, H,

            # def debug(attention_scores, selected_beam):
            #     b = 0
            #     print('selected_beam {}'.format(selected_beam[b]))
            #     print('attention_scores \n {}'.format( \
            #         [a[b, :, 0, :3] \
            #             for a in attention_scores]))
            # debug(attention_scores, selected_beam)
            # input()

            seq_logprob = selected_logprob.unsqueeze(-1)# B,beam_size,1
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1)) #B, beam_size, 1
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1)) #B, beam_size, 1

            this_word_logprob_on_beam = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1])) #B,bs,#V
            # print(this_word_logprob_on_beam.shape)
            # input()
            this_word_logprob = torch.gather(this_word_logprob_on_beam, 2, selected_words.unsqueeze(-1))#B,bs,
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            distributions = list(
                torch.gather(d, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, this_word_logprob_on_beam.shape[-1])) for d in distributions)
            distributions.append(this_word_logprob_on_beam)
            selected_words = selected_words.view(-1, 1)#B*beam_size 1
            wt = selected_words.squeeze(-1) #B*beam_size

            if t == 0:
                att_feats = utils.expand_tensor(att_feats, beam_size)
                gv_feat = utils.expand_tensor(gv_feat, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                p_att_feats = utils.expand_tensor(p_att_feats, beam_size)

                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1) #B,bs,1,L
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        distributions= torch.stack(distributions, -2) #B,bs,L,#V
        distributions = torch.gather(distributions, 1, sort_idxs.unsqueeze(-1).expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN, distributions.shape[-1]))
        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]
        distributions = distributions.contiguous()[:,0] # B,L,#V


        if output_attention:
            sort_idx_ex = sort_idxs.unsqueeze(-1).unsqueeze(-1) #B, bs, 1 -> B, bs, 1, 1, 1
            attention_scores = torch.stack(attention_scores, -1) #B,bs,H,N,T
            sort_idx_ex  = sort_idx_ex.expand(batch_size, beam_size, 
                attention_scores.shape[-3], attention_scores.shape[-2], cfg.MODEL.SEQ_LEN)
            #print(attention_scores.shape, sort_idx_ex.shape)
            attention_scores = torch.gather(attention_scores, 1, sort_idx_ex) #B, bs, H,N,T
            attention_scores = attention_scores.contiguous()[:,0] # B,H,N,T
            #print(attention_scores.shape, att_mask0.shape)
            #input()
            return outputs, log_probs, attention_scores, att_mask0,distributions
        else:
            return outputs, log_probs,distributions

    # For the experiments of X-LAN, we use the following beam search code, 
    # which achieves slightly better results but much slower.
    
    def decode_beam_xlan(self, **kwargs):
       beam_size = kwargs['BEAM_SIZE']
       gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
       batch_size = gv_feat.size(0)
    
       sents = Variable(torch.zeros((cfg.MODEL.SEQ_LEN, batch_size), dtype=torch.long).cuda())
       logprobs = Variable(torch.zeros(cfg.MODEL.SEQ_LEN, batch_size).cuda())   
       self.done_beams = [[] for _ in range(batch_size)]
       for n in range(batch_size):
           state = self.init_hidden(beam_size)
           gv_feat_beam = gv_feat[n:n+1].expand(beam_size, gv_feat.size(1)).contiguous()
           # bs, N 
           att_feats_beam = att_feats[n:n+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
           # bs, N, D
           att_mask_beam = att_mask[n:n+1].expand(*((beam_size,)+att_mask.size()[1:]))
           # bs, N
           p_att_feats_beam = p_att_feats[n:n+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous() if p_att_feats is not None else None
    
           wt = Variable(torch.zeros(beam_size, dtype=torch.long).cuda())
           kwargs = self.make_kwargs(wt, gv_feat_beam, att_feats_beam, att_mask_beam, p_att_feats_beam, state, **kwargs)
           logprobs_t, state = self.get_logprobs_state(**kwargs) 
           # logprobs_t  beam_size, #V
           # state [(#Layer, bs, D),(#Layer, bs, D)]
    
           self.done_beams[n] = self.beam_search(state, logprobs_t, **kwargs)
           sents[:, n] = self.done_beams[n][0]['seq'] 
           logprobs[:, n] = self.done_beams[n][0]['logps']
           #finally we choose the sentence with toppest !unaugmented! score
       return sents.transpose(0, 1), logprobs.transpose(0, 1)


    def decode(self, **kwargs):
        greedy_decode = kwargs['GREEDY_DECODE']
 
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)

        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs = self.make_kwargs(wt, gv_feat, att_feats, att_mask, p_att_feats, state)
            logprobs_t, state = self.get_logprobs_state(**kwargs)
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        return sents, logprobs
