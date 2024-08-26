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
ImgNet1k 










