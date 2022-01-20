#!/bin/bash
cd "${0%/*}/.." || exit

arch=$1
hostname=$(hostname)

case "$hostname" in
  "Server1")
    frames_dir=/media/data3/datasets/IPN/flow/flow
    train_annotation_file=/media/data3/datasets/IPN/annotations/Annot_TrainList.csv
    test_annotation_file=/media/data3/datasets/IPN/annotations/Annot_TestList.csv
    class_index_file=/media/data3/datasets/IPN/annotations/classIdx.csv
    device=cuda:0
    ;;
  "Server2")
    frames_dir=/mnt/disk3/datasets/IPN/flow/flow
    train_annotation_file=/mnt/disk3/datasets/IPN/annotations/Annot_TrainList.csv
    test_annotation_file=/mnt/disk3/datasets/IPN/annotations/Annot_TestList.csv
    class_index_file=/mnt/disk3/datasets/IPN/annotations/classIdx.csv
    device=cuda:1
    ;;
  "hungvuong")
    frames_dir=/ext_data2/comvis/datasets/IPN/flow/flow
    train_annotation_file=/ext_data2/comvis/datasets/IPN/annotations/Annot_TrainList.csv
    test_annotation_file=/ext_data2/comvis/datasets/IPN/annotations/Annot_TestList.csv
    class_index_file=/ext_data2/comvis/datasets/IPN/annotations/classIdx.csv
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
      frames_dir=/content/drive/MyDrive/datasets/IPN/flow/flow
      train_annotation_file=/content/drive/MyDrive/datasets/IPN/annotations/Annot_TrainList.csv
      test_annotation_file=/content/drive/MyDrive/datasets/IPN/annotations/Annot_TestList.csv
      class_index_file=/content/drive/MyDrive/datasets/IPN/annotations/classIdx.csv
      device=cuda:0
    else
      echo "[bash] Server $hostname is not supported!"
      exit 1
    fi
    ;;
esac
echo "[bash] Running on $hostname"

python3 train_ipn.py \
  --frames_dir $frames_dir \
  --train_annotation_file $train_annotation_file \
  --test_annotation_file $test_annotation_file \
  --class_index_file $class_index_file \
  --output_dir outputs/IPN/OF/R2plus1D \
  --temporal_slice 32 \
  --arch $arch \
  --max_epoch 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --device $device
#  --mean_std_file outputs/IPN/OF_mean_std.pt \
