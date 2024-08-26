import pandas as pd
import os
import numpy as np
from pprint import pprint
import torch
import sys
# import funcs
from collections import Counter


def get_target(label_filename, num_classes=1000):
    cls_idx = []
    with open(label_filename, 'r') as f:
        for line in f.readlines():
            segs = line.strip().split(' ')
            cls_idx.append(int(segs[-1]))
    cls_idx = np.array(cls_idx, dtype='int')
    label_stat = Counter(cls_idx)
    cls_num = [-1 for _ in range(num_classes)]
    for i in range(num_classes):
        cat_num = int(label_stat[i])
        cls_num[i] = cat_num
    target = cls_num / np.sum(cls_num)
    return target
    
def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1
        
    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95, threshold

def get_energy_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort(reversed=True)
    novel.sort(reversed=True)

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] > known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1
        
    fpr_at_tpr95 = np.sum(novel < threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95, threshold

def cal_ood_metrics(known, novel, method=None):

    """
    Note that the convention here is that ID samples should be labelled '1' and OoD samples should be labelled '0'
    Computes standard OoD-detection metrics: mtypes = ['FPR' (FPR @ TPR 95), 'AUROC', 'DTERR', 'AUIN', 'AUOUT']
    """

    tp, fp, fpr_at_tpr95, threshold = get_curve(known, novel, method)
    # get_energy_curve
    results = dict()
    
    mtype = 'Threshold'
    results[mtype] = threshold

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results

def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR

def get_osr_ood_metric(model, dataloader_id, dataloader_ood, method='Energy'):
    import funcs
    print(f'==> The test indicator is {method} !')
    # Get labels
        
    osr_labels = [1] * len(dataloader_id.dataset) + [0] * len(dataloader_ood.dataset)                  # 1 if sample is ID else 0
    osr_labels = np.array(osr_labels)
    id_labels = np.array([x[1] for x in dataloader_id.dataset.samples])

    # Get preds
    if method == 'MSP':
        id_preds, osr_preds_id_samples = funcs.test_predict_MSP(model, dataloader_id)
        _, osr_preds_osr_samples = funcs.test_predict_MSP(model, dataloader_ood)
    elif method == 'Energy':
        id_preds, osr_preds_id_samples = funcs.test_predict_Energy(model, dataloader_id)
        _, osr_preds_osr_samples = funcs.test_predict_Energy(model, dataloader_ood)
    elif method == 'GradNorm':
        id_preds, osr_preds_id_samples = funcs.test_predict_GradNorm(model, dataloader_id)
        _, osr_preds_osr_samples = funcs.test_predict_GradNorm(model, dataloader_ood)  
    elif method == 'ODIN':
        id_preds, osr_preds_id_samples = funcs.test_predict_ODIN(model, dataloader_id)
        _, osr_preds_osr_samples = funcs.test_predict_ODIN(model, dataloader_ood)  
    elif method == 'Energy_RW':
        target = get_target(r"./splits/val_labeled.txt")
        id_preds, osr_preds_id_samples = funcs.test_predict_Energy_RW(model, dataloader_id, target)
        _, osr_preds_osr_samples = funcs.test_predict_Energy_RW(model, dataloader_ood, target)
    elif method == 'MSP_RP':
        target = get_target(r"./splits/val_labeled.txt")
        id_preds, osr_preds_id_samples = funcs.test_predict_MSP_RP(model, dataloader_id, target)
        _, osr_preds_osr_samples = funcs.test_predict_MSP_RP(model, dataloader_ood, target)
    elif method == 'GradNorm_RP':
        target = get_target(r"./splits/val_labeled.txt")
        id_preds, osr_preds_id_samples = funcs.test_predict_GradNorm_RP(model, dataloader_id, target)
        _, osr_preds_osr_samples = funcs.test_predict_GradNorm_RP(model, dataloader_ood, target)
    elif method == 'ODIN_RW':
        target = get_target(r"./splits/val_labeled.txt")
        id_preds, osr_preds_id_samples = funcs.test_predict_ODIN_RW(model, dataloader_id, target)
        _, osr_preds_osr_samples = funcs.test_predict_ODIN_RW(model, dataloader_ood, target)               
    else:
        raise ValueError('Invalid method')

    # get metrics
    results = cal_ood_metrics(osr_preds_id_samples, osr_preds_osr_samples)         # Compute OoD metrics
    results['OSCR'] = compute_oscr(
        osr_preds_id_samples,
        osr_preds_osr_samples,
        id_preds,
        id_labels
    )
    results['ACC'] = (id_labels == id_preds).mean()

    return results

def get_osr_ood_metric_from_result(id_labels, id_preds, osr_preds_id_samples, osr_preds_osr_samples):
    # get metrics
    results = cal_ood_metrics(osr_preds_id_samples, osr_preds_osr_samples)         # Compute OoD metrics
    results['OSCR'] = compute_oscr(
        osr_preds_id_samples,
        osr_preds_osr_samples,
        id_preds,
        id_labels
    )
    results['ACC'] = (id_labels == id_preds).mean()

    return results

if __name__ == "__main__":
    import sys
    BASE_PATH = os.path.dirname(os.path.dirname(__file__))   # 文件路径一般会设置成常量
    sys.path.append(BASE_PATH)
    import data.dataset_osr_test
    import option
    import model.get_model
    import timm
    import test_option
    from torch.optim.swa_utils import AveragedModel
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    args = test_option.get_args_parser()
    # 加载数据
    imagenet_1k_root = r"/datassd/Inet1K/"
    imagenet_21k_root = r"/data/ImageNet-21K/"
    data_json_path = r"./splits/imagenet_ssb_splits.json"
    # label_filename = r"./splits/val_labeled.txt"
    
    model_path = r'ImgNet1k_out/baseline10/imagenet_fmfp-mixup_0.2-crl_0.0'
    act_threshold = 0.56828 # if use ReAct else 1000
    # method = 'Energy_RW' # MSP、Energy、ODIN、GradNorm or Energy_RW、MSP_RP、GradNorm_RP、ODIN_RW
    
    if args.method == 'GradNorm':
        batch_size = 1
    else:
        batch_size = 64
    dataloader_id, dataloader_ood = data.dataset_osr_test.get_dataload([0],imagenet_1k_root, imagenet_21k_root, batch_size, data_json_path=data_json_path)
    print(len(dataloader_id.dataset))
    print(len(dataloader_ood.dataset))
    
    # 加载模型
    net = model.get_model.get_model(1000, None, args)
    if args.optim_name == 'fmfp' or args.optim_name == 'swa':
        net = AveragedModel(net)
    net.load_state_dict(torch.load(os.path.join(model_path, f'best_acc_net.pth')), strict=True)
    net = net.cuda()
    net.module.head.threshold = args.act_threshold

    # # 预测并得到结果
    result = get_osr_ood_metric(net, dataloader_id, dataloader_ood, method=args.method)

    msg = f"==> The test indicator is {args.method}, using ReAct, threshold is {args.act_threshold:.4f}!\n"
    msg += "AUROC:{:.2f};  FPR:{:.2f};  OSCR:{:.2f};  ACC:{:.2f};  AUIN:{:.2f};  AUOUT:{:.2f};  DTERR:{:.2f};".format( \
            round(result['AUROC']*100, 2),  round(result['FPR']*100, 2), round(result['OSCR']*100, 2), round(result['ACC']*100, 2), \
            round(result['AUIN']*100, 2),  round(result['AUOUT']*100, 2), round(result['DTERR']*100, 2) )

    with open(os.path.join(model_path, 'test_result.txt'), 'a+') as f:
        f.write(msg + '\n')
    print(msg)


