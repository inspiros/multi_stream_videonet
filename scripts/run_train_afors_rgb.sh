#!/bin/bash
cd "${0%/*}/.." || exit

hostname=$(hostname)
echo "[bash] Running on $hostname"

case "$hostname" in
  "Server2")
    video_dir=/mnt/disk3/datasets/afors2022/data
    train_annotation_file=/mnt/disk3/datasets/afors2022/train.txt
    test_annotation_file=/mnt/disk3/datasets/afors2022/val.txt
    device=cuda:1
    ;;
  "hungvuong")
    video_dir=/ext_data2/comvis/datasets/afors2022/data
    train_annotation_file=/ext_data2/comvis/datasets/afors2022/train.txt
    test_annotation_file=/ext_data2/comvis/datasets/afors2022/val.txt
    device=cuda:3
    ;;
  *)
    echo "[bash] Server $hostname is not supported!"
    exit 1
    ;;
esac

python3 train_afors.py \
  --video_dir $video_dir \
  --train_annotation_file $train_annotation_file \
  --test_annotation_file $test_annotation_file \
  --output_dir outputs/AFORS/RGB/R3D \
  --max_epoch 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --device $device
#  --mean_std_file outputs/AFORS/RGB_mean_std.pt \
