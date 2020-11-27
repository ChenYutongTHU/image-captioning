import os
import sys
import tempfile
import json
from json import encoder
from lib.config import cfg

sys.path.append(cfg.INFERENCE.COCO_PATH)
from my_pycocotools.coco import COCO
from my_pycocoevalcap.eval import COCOEvalCap

class COCOEvaler(object):
    def __init__(self, annfile):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(annfile)
        if not os.path.exists(cfg.TEMP_DIR):
            os.makedirs(cfg.TEMP_DIR)

    def eval(self, result, imgToEval=True, SPICE=False):
        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=cfg.TEMP_DIR)
        json.dump(result, in_file) #[{‘image_id’: int, 'caption':string},{'image_id':int,'caption':string}]
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes, SPICE=SPICE)
        cocoEval.evaluate()
        os.remove(in_file.name)
        if imgToEval:
            return cocoEval.eval, cocoEval.imgToEval
        else:
            return cocoEval.eval