import pandas as pd
import numpy as np
import os
import data
from tqdm import tqdm
from utils.test_osr_ood import get_osr_ood_metric_from_result


def save_predictions_to_csv(image_names, id_preds, osr_preds, filename):
    data = {
        'img': image_names,
        'id_preds': id_preds,
        'osr_preds': osr_preds
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


def export_csv(id_image_name_path, ood_image_name_path, id_preds_cls, id_preds_score, ood_preds_score, csv_save_path):
    # get image_name
    with open(id_image_name_path, 'r') as fd:
        id_image_names = fd.readlines() 
        id_image_names = [name.strip() for name in id_image_names]
    
    with open(ood_image_name_path, 'r') as fd:
        ood_image_names = fd.readlines() 
        ood_image_names = [name.strip() for name in ood_image_names]   

    easy_ood_sample_len = 50000

    img_ID_new_names = []
    for img_ID_name in id_image_names:
        name_splits = img_ID_name.split("_")
        img_ID_new_name = name_splits[0] + "_" + name_splits[1] + "_" + name_splits[2] + os.path.splitext(img_ID_name)[1]
        img_ID_new_names.append(img_ID_new_name)
    id_image_names = np.array(img_ID_new_names)

    print("==========Exporting CSV FILE==========")
    image_ood_ID_index = np.ones_like(ood_preds_score)*-1
    all_image_names = np.concatenate([id_image_names, ood_image_names[:easy_ood_sample_len], id_image_names, ood_image_names[easy_ood_sample_len:]])
    all_id_preds = np.concatenate([id_preds_cls, image_ood_ID_index[:easy_ood_sample_len], id_preds_cls, image_ood_ID_index[easy_ood_sample_len:]])
    all_osr_preds = np.concatenate([id_preds_score, ood_preds_score[:easy_ood_sample_len], id_preds_score, ood_preds_score[easy_ood_sample_len:]])

    save_predictions_to_csv(all_image_names, all_id_preds, all_osr_preds, csv_save_path)



def get_predict_result(result_dir):

    id_preds_softmax = np.load(os.path.join(result_dir, "id_preds_softmax.npy"))
    id_preds_labels = np.load(os.path.join(result_dir, "id_preds_labels.npy"))
    id_preds_score = np.load(os.path.join(result_dir, "id_preds_score.npy"))
    ood_preds_score = np.load(os.path.join(result_dir, "ood_preds_score.npy"))

    return id_preds_softmax, id_preds_labels, id_preds_score, ood_preds_score


def get_all_predict_result(result_dirs):

    id_preds_softmax_list = []
    id_preds_score_list = []
    ood_preds_score_list = []
    for rdir in result_dirs:

        id_preds_softmax, id_preds_labels, id_preds_score, ood_preds_score = get_predict_result(rdir)
        id_preds_softmax_list.append(id_preds_softmax)
        id_preds_score_list.append(id_preds_score)
        ood_preds_score_list.append(ood_preds_score)
    
    return id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list


def calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score):

    result = get_osr_ood_metric_from_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)     

    msg = "AUROC:{:.2f};  FPR:{:.2f};  OSCR:{:.2f};  ACC:{:.2f};  AUIN:{:.2f};  AUOUT:{:.2f};  DTERR:{:.2f};".format( \
            round(result['AUROC']*100, 2),  round(result['FPR']*100, 2), round(result['OSCR']*100, 2), round(result['ACC']*100, 2), \
            round(result['AUIN']*100, 2),  round(result['AUOUT']*100, 2), round(result['DTERR']*100, 2) )

    print(msg)


def metric_single_result_GradNorm(result_dir):
    id_preds_softmax, id_preds_labels, id_preds_score, ood_preds_score = get_predict_result(result_dir)

    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)


def metric_ave_result_GradNorm(result_dirs):

    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    num_result = len(id_preds_softmax_list)
    id_preds_softmax = np.zeros_like(id_preds_softmax_list[0])
    id_preds_score = np.zeros_like(id_preds_score_list[0])
    ood_preds_score = np.zeros_like(ood_preds_score_list[0])
    for idx in range(num_result):
        id_preds_softmax += id_preds_softmax_list[idx]
        id_preds_score += id_preds_score_list[idx]
        ood_preds_score += ood_preds_score_list[idx]
    
    id_preds_softmax = id_preds_softmax/num_result
    id_preds_score = id_preds_score/num_result
    ood_preds_score = ood_preds_score/num_result
    
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)

    return id_preds_cls, id_preds_score, ood_preds_score


def metric_ave_weight_tencrop_result_GradNorm(result_dirs, weight):

    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    num_result = len(id_preds_softmax_list)

    id_preds_softmax = np.zeros_like(id_preds_softmax_list[0])
    id_preds_score = np.zeros_like(id_preds_score_list[0])
    ood_preds_score = np.zeros_like(ood_preds_score_list[0])
    for idx in range(num_result):
        if idx == 0 or idx == 5:
            ratio = weight
        else:
            ratio = 1
        id_preds_softmax += id_preds_softmax_list[idx]*ratio
        id_preds_score += id_preds_score_list[idx]*ratio
        ood_preds_score += ood_preds_score_list[idx]*ratio
    
    id_preds_softmax = id_preds_softmax/num_result
    id_preds_score = id_preds_score/num_result
    ood_preds_score = ood_preds_score/num_result
    
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)

    return id_preds_cls, id_preds_score, ood_preds_score



def metric_ave_remove_maxnum_result_GradNorm(result_dirs, idx):

    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    num_result = len(id_preds_softmax_list)

    id_preds_softmax = np.zeros_like(id_preds_softmax_list[0])
    id_preds_score = np.zeros_like(id_preds_score_list[0])
    ood_preds_score = np.zeros_like(ood_preds_score_list[0])

    id_preds_softmax = np.sum(np.sort(np.array(id_preds_softmax_list), axis=0)[idx:], axis=0)/(num_result-idx)
    id_preds_score = np.sum(np.sort(np.array(id_preds_score_list), axis=0)[idx:], axis=0)/(num_result-idx)
    ood_preds_score = np.sum(np.sort(np.array(ood_preds_score_list), axis=0)[idx:], axis=0)/(num_result-idx)
    
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)


def metric_ave_sub_std_result_GradNorm(result_dirs, ratio):

    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    num_result = len(id_preds_softmax_list)
    id_preds_softmax_mean = np.mean(np.array(id_preds_softmax_list), axis=0)
    id_preds_score_mean = np.mean(np.array(id_preds_score_list), axis=0)
    ood_preds_score_mean = np.mean(np.array(ood_preds_score_list), axis=0)

    id_preds_softmax_std = np.std(np.array(id_preds_softmax_list), axis=0)
    id_preds_score_std = np.std(np.array(id_preds_score_list), axis=0)
    ood_preds_score_std = np.std(np.array(ood_preds_score_list), axis=0)

    # id_preds_softmax = id_preds_softmax/num_result
    id_preds_softmax = id_preds_softmax_mean - ratio*id_preds_softmax_std
    id_preds_score = id_preds_score_mean - ratio*id_preds_score_std
    ood_preds_score = ood_preds_score_mean - ratio*ood_preds_score_std
    
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)



def metric_multi_result_GradNorm(result_dirs):

    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    # print(f"id_preds_softmax_list: {id_preds_softmax_list}")
    # print(f"id_preds_softmax_list: {id_preds_softmax_list}")
    num_result = len(id_preds_softmax_list)

    id_preds_softmax = np.ones_like(id_preds_softmax_list[0])
    id_preds_score = np.ones_like(id_preds_score_list[0])
    ood_preds_score = np.ones_like(ood_preds_score_list[0])
    for idx in range(num_result):
        id_preds_softmax *= id_preds_softmax_list[idx]
        id_preds_score *= id_preds_score_list[idx]
        ood_preds_score *= ood_preds_score_list[idx]
    
    id_preds_softmax = id_preds_softmax ** (1/num_result)
    id_preds_score = id_preds_score ** (1/num_result)
    ood_preds_score = ood_preds_score ** (1/num_result)
    
    # 对结果进行处理
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)


def metric_temperature_sharpen_result_GradNorm(result_dirs, temperature):

    # 将数据全部读取出来
    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    # print(f"id_preds_softmax_list: {id_preds_softmax_list}")
    # print(f"id_preds_softmax_list: {id_preds_softmax_list}")
    num_result = len(id_preds_softmax_list)

    id_preds_softmax = np.ones_like(id_preds_softmax_list[0])
    id_preds_score = np.ones_like(id_preds_score_list[0])
    ood_preds_score = np.ones_like(ood_preds_score_list[0])
    for idx in range(num_result):
        id_preds_softmax += id_preds_softmax_list[idx] ** temperature
        id_preds_score += id_preds_score_list[idx] ** temperature
        ood_preds_score += ood_preds_score_list[idx] ** temperature
    
    id_preds_softmax = id_preds_softmax/num_result
    id_preds_score = id_preds_score/num_result
    ood_preds_score = ood_preds_score/num_result
    
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)



def metric_simi_result_GradNorm(result_dirs):

    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    num_result = len(id_preds_softmax_list)

    id_preds_softmax = np.zeros_like(id_preds_softmax_list[0])
    id_preds_score = np.zeros_like(id_preds_score_list[0])
    ood_preds_score = np.zeros_like(ood_preds_score_list[0])
    count = 0
    for idx in range(num_result):
        if idx == (num_result - 1):
            break
        for jdx in range(idx+1, num_result):
            id_preds_softmax += id_preds_softmax_list[idx] * id_preds_softmax_list[jdx]
            id_preds_score += id_preds_score_list[idx] * id_preds_score_list[jdx]
            ood_preds_score += ood_preds_score_list[idx] * ood_preds_score_list[jdx]
            count += 1
    id_preds_softmax = id_preds_softmax/count
    id_preds_score = id_preds_score/count
    ood_preds_score = ood_preds_score/count
    
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)


def metric_simi_result_GradNorm_idx_src(result_dirs, idx_map):

    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    # print(f"id_preds_softmax_list: {id_preds_softmax_list}")
    # print(f"id_preds_softmax_list: {id_preds_softmax_list}")
    num_result = len(id_preds_softmax_list)

    id_preds_softmax = np.zeros_like(id_preds_softmax_list[0])
    id_preds_score = np.zeros_like(id_preds_score_list[0])
    ood_preds_score = np.zeros_like(ood_preds_score_list[0])
    count = 0
    for idx in range(num_result):
        if idx == idx_map:
            continue
        id_preds_softmax += id_preds_softmax_list[idx] * id_preds_softmax_list[idx_map]
        id_preds_score += id_preds_score_list[idx] * id_preds_score_list[idx_map]
        ood_preds_score += ood_preds_score_list[idx] * ood_preds_score_list[idx_map]
        count += 1

    id_preds_softmax = id_preds_softmax/count
    id_preds_score = id_preds_score/count
    ood_preds_score = ood_preds_score/count
    
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)


def metric_max_GradNorm_and_ave_acc(result_dirs):

    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    num_result = len(id_preds_softmax_list)

    id_preds_softmax = np.zeros_like(id_preds_softmax_list[0])
    id_preds_score = np.zeros_like(id_preds_score_list[0])
    ood_preds_score = np.zeros_like(ood_preds_score_list[0])
    for idx in range(num_result):
        id_preds_softmax += id_preds_softmax_list[idx]
    
    id_preds_softmax = id_preds_softmax/num_result
    id_preds_score = np.max(np.array(id_preds_score_list), axis=0)
    ood_preds_score = np.max(np.array(ood_preds_score_list), axis=0)
    
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)


def metric_min_GradNorm_and_ave_acc(result_dirs):

    id_preds_softmax_list, id_preds_labels, id_preds_score_list, ood_preds_score_list = get_all_predict_result(result_dirs)

    num_result = len(id_preds_softmax_list)

    id_preds_softmax = np.zeros_like(id_preds_softmax_list[0])
    id_preds_score = np.zeros_like(id_preds_score_list[0])
    ood_preds_score = np.zeros_like(ood_preds_score_list[0])
    for idx in range(num_result):
        id_preds_softmax += id_preds_softmax_list[idx]
    
    id_preds_softmax = id_preds_softmax/num_result
    id_preds_score = np.min(np.array(id_preds_score_list), axis=0)
    ood_preds_score = np.min(np.array(ood_preds_score_list), axis=0)
    
    id_preds_cls = np.argmax(id_preds_softmax, axis=-1)

    calculate_result(id_preds_labels, id_preds_cls, id_preds_score, ood_preds_score)
   


if __name__ == "__main__":
    import argparse
    from thop import profile
    parser = argparse.ArgumentParser(description="List subdirectories in the specified result directory.")

    parser.add_argument(
        '--result_dir', 
        type=str, 
        default=r"exp/2_baseline_res384ep30_inputsize384_ReAct1.5/10c", 
        help="Path to the result directory"
    )
    
    
    args = parser.parse_args()
    result_dir = args.result_dir
    sub_dirs = sorted(os.listdir(result_dir))
    print(sub_dirs)

    sub_dir_list = []

    for i, sdir in enumerate(sub_dirs):
        if i < 5:
            sub_dir_list.append(os.path.join(result_dir, sdir))
       
    id_preds_cls, id_preds_score, ood_preds_score = metric_ave_result_GradNorm(sub_dir_list)

    id_image_name_path = os.path.join(result_dir, sub_dirs[0], "id_image_name.txt")
    ood_image_name_path = os.path.join(result_dir, sub_dirs[0], "ood_image_name.txt")
    csv_save_path = os.path.join(result_dir, sub_dirs[0], "result.csv")
    export_csv(id_image_name_path, ood_image_name_path, id_preds_cls, id_preds_score, ood_preds_score, csv_save_path)



