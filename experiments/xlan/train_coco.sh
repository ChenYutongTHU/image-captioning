str='1,2,3,4,5,6'
export CUDA_VISIBLE_DEVICES=$str
len=${#CUDA_VISIBLE_DEVICES}
val=`expr ${len} + 1`
val2=`expr $val / 2`
python3 -m torch.distributed.launch --nproc_per_node=$val2 --master_port=29503  main.py \
    --folder ./experiments/xlan/train_coco --config ./experiments/xlan/config_coco.yml \
    --dataset_name coco \
    --summary_freq_scalar=1000 --summary_freq_img2cap=1000 \
