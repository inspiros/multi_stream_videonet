#!/bin/bash
cd "${0%/*}/.." || exit

arch=$1
hostname=$(hostname)

case "$hostname" in
  "Server1")
    video_dir=/media/data3/datasets/afors2022/data
    train_annotation_file=/media/data3/datasets/afors2022/train.txt
    test_annotation_file=/media/data3/datasets/afors2022/val.txt
    device=cuda:0
    ;;
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
    is_colab=false
    for dir in /usr/local/lib/python*/dist-packages/google/colab;
    do
      if [ -d "$dir" ];
      then
        is_colab=true
      fi
    done

    if [ $is_colab = true ];
    then  # run on google colab
      hostname="[colab]$hostname"
      video_dir=/content/drive/MyDrive/datasets/afors2022/data
      train_annotation_file=/content/drive/MyDrive/datasets/afors2022/train.txt
      test_annotation_file=/content/drive/MyDrive/datasets/afors2022/val.txt
      device=cuda:0
    else
      echo "[bash] Server $hostname is not supported!"
      exit 1
    fi
    ;;
esac
echo "[bash] Running on $hostname"

python3 train_afors.py \
  --video_dir $video_dir \
  --train_annotation_file $train_annotation_file \
  --test_annotation_file $test_annotation_file \
  --output_dir outputs/AFORS/RGB/R2plus1D \
  --arch $arch \
  --max_epoch 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --device $device
#  --mean_std_file outputs/AFORS/RGB_mean_std.pt \
