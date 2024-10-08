### Baseline
### SURE
NCCL_P2P_DISABLE=1 python3 main.py \
--batch-size 32 \
--gpu 8 9  \
--epochs 20 \
--lr 1e-2 \
--weight-decay 5e-3 \
--swa-epoch-start 0 \
--swa-lr 1e-4 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0.2 \
--mixup-beta 10 \
--use-cosine \
--save-dir exp/demo \
--deit-path ./deit_3_base_384_mod.pth \
ImgNet1k