import os, json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='./X-LAN-models-results/xlan/result', help='directory containing result_val/test*.json')
    parser.add_argument('--split', default='test')
    parser.add_argument('--epoch', type=int, default=47, help='')
    parser.add_argument('--img_rootdir', default='./karpathy_test5k/')
    parser.add_argument('--imgname_file', default='./coco_info/id2name_123287.json')
    parser.add_argument('--GT_file', default='./coco_info/GTcaptions_COCOval_40506.json')


    args = parser.parse_args()
    with open(os.path.join(args.directory,'result_{}_{}.json'.format(args.split,args.epoch)), 'r') as f:
        result = json.load(f) #list of dict
    with open(args.imgname_file, 'r') as f:
        id2name = json.load(f)
    with open(args.GT_file, 'r') as f:
        GT_caption = json.load(f)

    result2 = []
    for res0 in result:
        id_ = str(res0['image_id'])
        res = {}
        res['coco_file'] = os.path.join(args.img_rootdir, 'coco', id2name[id_])
        res['peteranderson_file'] = os.path.join(args.img_rootdir, 'peteranderson', id2name[id_])
        res['pred_caption'] = res0['caption']
        for i in range(5):
            res['GT_caption{}'.format(i)] = GT_caption[id_][i]
        result2.append(res)

    with open(os.path.join(args.directory, 'result2_{}_{}.json'.format(args.split, args.epoch)), 'w') as f:
        json.dump(result2,f)

