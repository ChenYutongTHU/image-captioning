str='1,2,3,4,5,6'
export CUDA_VISIBLE_DEVICES=$str
len=${#CUDA_VISIBLE_DEVICES}
val=`expr ${len} + 1`
val2=`expr $val / 2`
python3 -m torch.distributed.launch --nproc_per_node=$val2 --master_port=29501 main.py \
    --folder ./experiments/xlan/train_aic_coco --config ./experiments/xlan/config.yml \
    --dataset_name=coco_aic \
    --summary_freq_scalar=100 --summary_freq_img2cap=5000 

