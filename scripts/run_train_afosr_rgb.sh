#!/bin/bash
cd "${0%/*}/.." || exit

host=$(hostname)
echo "[sh] Running on $host"

if ((host == "Server2")); then
  video_dir=/mnt/disk3/datasets/afosr2022/data
  train_annotation_file=/mnt/disk3/datasets/afosr2022/train.txt
  test_annotation_file=/mnt/disk3/datasets/afosr2022/val.txt
  device=cuda:1
elif ((host == "hungvuong")); then
  video_dir=/ext_data2/comvis/datasets/afosr2022/data
  train_annotation_file=/ext_data2/comvis/datasets/afosr2022/train.txt
  test_annotation_file=/ext_data2/comvis/datasets/afosr2022/val.txt
  device=cuda:3
else
  echo "Host not supported!"
  exit 1
fi

python3 train_afosr.py \
  --video_dir $video_dir \
  --train_annotation_file $train_annotation_file \
  --test_annotation_file $test_annotation_file \
  --output_dir outputs/AFOSR/RGB \
  --max_epoch 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --device $device
#  --mean_std_file outputs/AFOSR/RGB_mean_std.pt \
