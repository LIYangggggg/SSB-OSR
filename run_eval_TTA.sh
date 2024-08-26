### Baseline
### SURE
CUDA_VISIBLE_DEVICES=0 \
python3 test_osr_ood_TTA.py \
--batch-size 128 \
--mode '5c' \
--input-size 480 \
--save-dir  ./exp \
--use-cosine \
--cos-temp 8 \
--save-pth ./best_acc_net.pth \
ImgNet1k \
--train-dir /datassd/Inet1K/train \
--val-dir /datassd/Inet1K/val \
--test-dir /datassd/Inet1K/val \
--id-dir /datassd/Inet1K/ \
--ood-dir /data/ImageNet-21K/ 










