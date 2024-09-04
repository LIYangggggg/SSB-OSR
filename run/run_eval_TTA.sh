### Baseline
### SURE
CUDA_VISIBLE_DEVICES=7 \
python3 test_osr_ood_TTA.py \
--batch-size 56 \
--mode '5c' \
--input-size 480 \
--save-dir  ./exp/best \
--use-cosine \
--cos-temp 8 \
--save-pth /data8022/Private/wushengliang/project_program/SSB_OSR_clone/emb_models/2_baseline_res384ep30/best_acc_net.pth \
ImgNet1k 










