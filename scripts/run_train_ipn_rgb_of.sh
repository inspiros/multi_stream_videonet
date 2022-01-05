cd "${0%/*}/.." || exit
python3 train_ipn_multi_stream.py \
  --frames_dirs /mnt/disk3/datasets/IPN/frames/frames /mnt/disk3/datasets/IPN/flow/flow \
  --train_annotation_file /mnt/disk3/datasets/IPN/annotations/Annot_TrainList.csv \
  --test_annotation_file /mnt/disk3/datasets/IPN/annotations/Annot_TestList.csv \
  --class_index_file /mnt/disk3/datasets/IPN/annotations/classIdx.csv \
  --output_dir outputs/IPN/RGB_OF \
  --max_epoch 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --device cuda:1
#  --mean_std_file outputs/IPN/RGB_OF_mean_std.pt \
