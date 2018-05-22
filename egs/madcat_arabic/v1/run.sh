<<<<<<< HEAD
#!/usr/bin/env bash

set -e
stage=0
=======
#!/bin/bash

set -e # exit on error
. ./path.sh

stage=0

. ./scripts/parse_options.sh # e.g. this parses the --stage option if supplied.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

>>>>>>> waldo-seg/master
nj=70
download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data
<<<<<<< HEAD

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. parse_options.sh  # e.g. this parses the above options
                      # if supplied.

./local/check_tools.sh

mkdir -p data/{train,test,dev}/data
mkdir -p data/local/{train,test,dev}
mkdir -p data/{train,test,dev}/masks
data_splits_dir=data/download/data_splits

if [ $stage -le 0 ]; then
  echo "$0: Downloading data splits..."
  echo "Date: $(date)."
  local/download_data.sh --data_splits $data_splits_dir
fi

if [ $stage -le 0 ]; then
  for dataset in test dev train; do
    echo "$0: Extracting mask from page image for dataset:  $dataset. "
    echo "Date: $(date)."
    dataset_file=$data_splits_dir/madcat.$dataset.raw.lineid
    local/extract_masks.sh --nj $nj --cmd $cmd --dataset_file $dataset_file \
                           --download_dir1 $download_dir1 --download_dir2 $download_dir2 \
                           --download_dir3 $download_dir3 data/local/$dataset
  done
=======
writing_condition1=/export/corpora/LDC/LDC2012T15/docs/writing_conditions.tab
writing_condition2=/export/corpora/LDC/LDC2013T09/docs/writing_conditions.tab
writing_condition3=/export/corpora/LDC/LDC2013T15/docs/writing_conditions.tab
local/check_dependencies.sh


if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --download_dir1 $download_dir1 --download_dir2 $download_dir2 \
      --download_dir3 $download_dir3 --writing_condition1 $writing_condition1 \
      --writing_condition2 $writing_condition2 --writing_condition3 $writing_condition3
fi


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
    --train-dir data \
    --train-image-size 128 \
    --core-config $dir/configs/core.config \
    --unet-config $dir/configs/unet.config \
    $dir/model_best.pth.tar

>>>>>>> waldo-seg/master
fi

