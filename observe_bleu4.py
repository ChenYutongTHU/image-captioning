import numpy as np 
import json
root = '/data/disk1/private/chenyutong/IC/image-captioning_xlan/experiments/xlan/train_aic_coco_warmup10k/result/'
with open(root+'coco+result_test_58_scores.json','r') as f:
    model1_scores = json.load(f)
with open(root+'coco+result_test_58.json','r') as f:
    model1_captions = json.load(f)

root = '/data/disk1/private/chenyutong/IC/image-captioning_xlan/experiments/xlan/train_coco_warmup10k/result/'
with open(root+'coco+result_test_57_scores.json','r') as f:
    model0_scores = json.load(f)
with open(root+'coco+result_test_57.json','r') as f:
    model0_captions = json.load(f)


with open('/data/disk1/private/FXData/COCO/annotations/id2references_val.json', 'r') as f:
    id2ref = json.load(f)

outf = open('COCO_cmp.json','w')
cnt = 0
for item0, item1 in zip(model0_captions, model1_captions):
    assert item0['image_id'] == item1['image_id']
    id_ = str(item0['image_id'])
    ref = id2ref[id_]
    cap0,cap1 = item0['caption'], item1['caption']
    score0, score1 = model0_scores[id_]['Bleu_4'],model1_scores[id_]['Bleu_4']
    if score0<0.01 and score1<0.01:
        outf.writelines('{}\n'.format(id_))
        outf.writelines('model0 {}   {}\n'.format(cap0, score0))
        outf.writelines('model1 {}   {}\n'.format(cap1, score1))
        for r in id2ref[id_]:
            outf.writelines('{}\n'.format(r))
        outf.writelines('\n')
        cnt += 1
    if cnt>=100:
        break