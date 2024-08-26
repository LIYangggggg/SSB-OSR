### Baseline
### SURE
python3 main.py \
--batch-size 128 \
--gpu 1 2 3 4 5 6 7 8 \
--epochs 20 \
--lr 0.01 \
--weight-decay 5e-5 \
--swa-epoch-start 0 \
--swa-lr 0.004 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0.2 \
--mixup-beta 10 \
--save-dir ./ImgNet1k_out/deit_out_0719 \
--finetune ./deit_3_base_384_1k.pth \
ImgNet1k \
--train-dir /datassd/Inet1K/train \
--val-dir /datassd/Inet1K/val \
--test-dir /datassd/Inet1K/val \
--id-dir /datassd/Inet1K/ \
--ood-dir /data/ImageNet-21K/ 
