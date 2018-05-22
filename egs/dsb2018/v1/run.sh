#!/bin/bash

set -e # exit on error
. ./path.sh

stage=0

<<<<<<< HEAD
# train/validate split
train_prop=0.9
seed=0

# network training setting
gpu_id=0
depth=5
epochs=10
height=128
width=128
batch=16
=======
. parse_options.sh  # e.g. this parses the --stage option if supplied.

>>>>>>> waldo-seg/master

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

<<<<<<< HEAD
. parse_options.sh  # e.g. this parses the --stage option if supplied.


local/check_dependencies.sh

=======
local/check_dependencies.sh


# train/validate split
train_prop=0.9
seed=0
>>>>>>> waldo-seg/master
if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --train_prop $train_prop --seed $seed
fi

<<<<<<< HEAD
name=unet_${depth}_${epochs}_sgd
if [ $stage -le 1 ]; then
  # training the network
  logdir=exp/$name
  mkdir -p $logdir
  $cmd --gpu 1 --mem 2G $logdir/train.log limit_num_gpus.sh local/train.py \
            --name $name \
            --depth $depth \
            --batch-size $batch \
            --img-height $height \
            --img-width $width \
            --epochs $epochs
=======

epochs=10
depth=5
dir=exp/unet_${depth}_${epochs}_sgd
if [ $stage -le 1 ]; then
  # training
  local/run_unet.sh --epochs $epochs --depth $depth
fi

if [ $stage -le 2 ]; then
    echo "doing segmentation...."
  local/segment.py \
    --dir $dir \
    --train-dir data/train_val \
    --train-image-size 128 \
    --core-config $dir/configs/core.config \
    --unet-config $dir/configs/unet.config \
    $dir/model_best.pth.tar

>>>>>>> waldo-seg/master
fi
