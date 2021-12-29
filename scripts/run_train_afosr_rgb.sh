cd "${0%/*}/.." || exit
python3 train_afosr.py \
  --video_dir /mnt/disk3/datasets/afosr2022/data \
  --train_annotation_file /mnt/disk3/datasets/afosr2022/train.txt \
  --test_annotation_file /mnt/disk3/datasets/afosr2022/val.txt \
  --output_dir outputs/AFOSR/RGB \
  --max_epoch 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --device cuda:1
#  --mean_std_file outputs/AFOSR/RGB_mean_std.pt \
