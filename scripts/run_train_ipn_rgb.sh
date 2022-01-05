cd "${0%/*}/.." || exit
python3 train_ipn.py \
  --frames_dir /mnt/disk3/datasets/IPN/frames/frames \
  --train_annotation_file /mnt/disk3/datasets/IPN/annotations/Annot_TrainList.csv \
  --test_annotation_file /mnt/disk3/datasets/IPN/annotations/Annot_TestList.csv \
  --class_index_file /mnt/disk3/datasets/IPN/annotations/classIdx.csv \
  --output_dir outputs/IPN/RGB \
  --temporal_slice 32 \
  --max_epoch 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --device cuda:1
#  --mean_std_file outputs/IPN/RGB_mean_std.pt \
