import pandas as pd
import numpy as np
import data.dataset_osr_test
from utils.test_osr_ood import get_osr_ood_metric_from_result


model_path = r'/data/Private/wushengliang/project_program/SSB_OSR_clone/res384ep30_use_trainlabel_2/10c/0/result.csv'
imagenet_1k_root = r"/data/Inet1K/"
imagenet_21k_root = r"/data/ImageNet-21K/"
data_json_path = r"./splits/imagenet_ssb_splits.json"


# 读取数据，获取数据标签
dataloader_id, dataloader_ood = data.dataset_osr_test.get_dataload([0],imagenet_1k_root, imagenet_21k_root, 64, data_json_path=data_json_path)
data_id_len = len(dataloader_id.dataset)
data_ood_len = len(dataloader_ood.dataset)
data_ood_easy_len = 50000

# 获取csv结果
df = pd.read_csv(model_path)
print(df.values.shape)
data_csv = df.values

id_preds = data_csv[0:data_id_len, 1]
osr_preds_id_samples = data_csv[0:data_id_len, 2]

osr_preds_osr_samples_easy = data_csv[data_id_len:data_id_len+data_ood_easy_len, 2]
osr_preds_osr_samples_hard = data_csv[data_id_len*2 + data_ood_easy_len:, 2]

osr_preds_osr_samples = np.concatenate((osr_preds_osr_samples_easy, osr_preds_osr_samples_hard), axis=0)

# Get labels
osr_labels = [1] * len(dataloader_id.dataset)*2 + [0] * data_ood_len               
osr_labels = np.array(osr_labels)
id_labels = np.array([x[1] for x in dataloader_id.dataset.samples])

# double id lable and pred
id_labels = np.concatenate((id_labels, id_labels), axis=0)
id_preds = np.concatenate((id_preds, id_preds), axis=0)
osr_preds_id_samples = np.concatenate((osr_preds_id_samples, osr_preds_id_samples), axis=0)

# 进行统计
result = get_osr_ood_metric_from_result(id_labels, id_preds, osr_preds_id_samples, osr_preds_osr_samples)

msg = "AUROC:{:.2f};  FPR:{:.2f};  OSCR:{:.2f};  ACC:{:.2f};  AUIN:{:.2f};  AUOUT:{:.2f};  DTERR:{:.2f};".format( \
        round(result['AUROC']*100, 2),  round(result['FPR']*100, 2), round(result['OSCR']*100, 2), round(result['ACC']*100, 2), \
        round(result['AUIN']*100, 2),  round(result['AUOUT']*100, 2), round(result['DTERR']*100, 2) )

print(msg)

