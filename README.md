# Introduction
This repository is for **X-Linear Attention Networks for Image Captioning** (CVPR 2020). The original paper can be found [here](https://arxiv.org/pdf/2003.14080.pdf).

Please cite with the following BibTeX:

```
@inproceedings{xlinear2020cvpr,
  title={X-Linear Attention Networks for Image Captioning},
  author={Pan, Yingwei and Yao, Ting and Li, Yehao and Mei, Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

<p align="center">
  <img src="images/framework.jpg" width="800"/>
</p>


## Requirements
* Python 3
* CUDA 10
* numpy
* tqdm
* easydict
* [PyTorch](http://pytorch.org/) (>1.0)
* [torchvision](http://pytorch.org/)
* [coco-caption](https://github.com/ruotianluo/coco-caption)

## Data preparation
1. Download [coco-caption](https://github.com/ruotianluo/coco-caption), setup the path of __C.INFERENCE.COCO_PATH in lib/config.py and **modify the directory name 'pycocoevalcap','pycocotools' to 'my_pycocoevalcap','my_pycocotools'**.

## Training
Only X-LAN model can be trained now :P.
### Train X-LAN model on COCO only
```
bash experiments/xlan/train_coco.sh
```
### Train X-LAN model on both AIC and COCO 
```
bash experiments/xlan/train_aic_coco.sh
```
## Tensorboard visualization
To view curves of training loss and evaluation scores
```
cd ./experiments/xlan
tensorboard --logdir='./' --port=6006
```

## Evaluation
### COCO & AIC testing set
```
export TRAINING_DIR='/data/disk1/private/FXData/Xlinear_models/train_coco_warmup10k'
CUDA_VISIBIE_DEVICES=0 python3 main_test.py \
--folder=$TRAINING_DIR --config=config_coco_warmup10k.yml --resume 57
```
Running this command will output 
1. caption results as json file in '$TRAINING_DIR/result'
2. Multi-headed attention visualization in '$TRAINING_DIR/result/attention_visualization' for the first 10 images in the testing set.
3. Testing scores of Bleu1-4, METEOR, ROUGE and CIDER, written in '$TRAINING_DIR/log.txt'. (SPICE is turned off to save computational cost)

To visualize predicted result paired with input image, 
```
cd /data/disk1/private/FXData/visualize

export DIR='${TRAINING_DIR}/result'
export FILE='coco+result_test_57.json'
python generate_result.py \
--directory=$DIR \
--result_file=$FILE \
--processedimg_dir=./vg_images \
--img_dir=./images \
--imgname_file=/data/disk1/private/FXData/COCO/coco_info/id2name_123287.json \
--GT_file=/data/disk1/private/FXData/COCO/coco_info/GTcaptions_COCOval_40506.json

python -m http.server 8009 --directory ./coco
```
See more details in '/data/disk1/private/FXData/visualize/visualize_coco.sh' and '/data/disk1/private/FXData/visualize/visualize_aic.sh'.

### Raw images as input
To generate caption for raw images, place all raw images in './test' directory and run the following commands
```
export CUDA_VISIBLE_DEVICES='1'
export BASE_DIR='the path to image-captioning repository'
export VG_DIR='/data/disk1/private/FXData/VG/'
export TEST_DIR="${BASE_DIR}/test"
export MODEL_DIR='/data/disk1/private/FXData/Xlinear_models/train_coco_warmup10k'
#model in MODEL_DIR/snapshot
cd $VG_DIR
python3 feature_api_yutong.py --img_folder="${TEST_DIR}/images" \
 --output_dir="${TEST_DIR}/vg"

cd $BASE_DIR
python3 main_test.py \
--test_raw_image \
--folder=$MODEL_DIR --resume 57 --config="${MODEL_DIR}/config_coco_warmup10k.yml" 
```
Then all results are output in './test' similarly as for COCO/AIC test images.


## Acknowledgements
Thanks the contribution of [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) and awesome PyTorch team.
