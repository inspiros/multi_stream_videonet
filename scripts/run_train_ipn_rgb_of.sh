#!/bin/bash
cd "${0%/*}/.." || exit

hostname=$(hostname)
echo "[bash] Running on $hostname"

case "$hostname" in
  "Server2")
    frames_dirs="/mnt/disk3/datasets/IPN/frames/frames /mnt/disk3/datasets/IPN/flow/flow"
    train_annotation_file=/mnt/disk3/datasets/IPN/annotations/Annot_TrainList.csv
    test_annotation_file=/mnt/disk3/datasets/IPN/annotations/Annot_TestList.csv
    class_index_file=/mnt/disk3/datasets/IPN/annotations/classIdx.csv
    device=cuda:1
    ;;
  "hungvuong")
    frames_dirs="/ext_data2/comvis/datasets/IPN/frames/frames /ext_data2/comvis/datasets/IPN/flow/flow"
    train_annotation_file=/ext_data2/comvis/datasets/IPN/annotations/Annot_TrainList.csv
    test_annotation_file=/ext_data2/comvis/datasets/IPN/annotations/Annot_TestList.csv
    class_index_file=/ext_data2/comvis/datasets/IPN/annotations/classIdx.csv
    device=cuda:3
    ;;
  *)
    echo "[bash] Server $hostname is not supported!"
    exit 1
    ;;
esac

python3 train_ipn_multi_stream.py \
  --frames_dirs $frames_dirs \
  --train_annotation_file $train_annotation_file \
  --test_annotation_file $test_annotation_file \
  --class_index_file $class_index_file \
  --output_dir outputs/IPN/RGB_OF \
  --temporal_slice 32 \
  --max_epoch 30 \
  --batch_size 4 \
  --lr 1e-3 \
  --device $device
#  --mean_std_file outputs/IPN/RGB_OF_mean_std.pt \
