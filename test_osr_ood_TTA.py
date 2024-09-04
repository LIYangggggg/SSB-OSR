import os
import torch
import torchvision.transforms
import data.dataset_osr_test
import model.get_model
import utils.test_option

import numpy as np
import torch.nn as nn

from copy import deepcopy
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.autograd import Variable

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_target(dataset_root, num_classes=1000):
    cls_num = np.zeros(num_classes, dtype=int)
    class_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]
    for class_index, folder in enumerate(class_folders):
        class_path = os.path.join(dataset_root, folder)
        num_images = len([img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))])
        cls_num[class_index] = num_images
    target = cls_num / np.sum(cls_num)
    return target

@torch.no_grad()
def test_predict_GradNorm_RP(model, test_loader, targets, num_classes=1000):
    """
    Get class predictions and Grad Norm Score for all instances in loader
    """

    model.eval()
    id_preds = []       # Store class preds
    gradnorm_preds = []      # Stores OSR preds
    save_labels = []
    image_names = []
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    feat_model = deepcopy(model)
    feat_model.module.head = nn.Sequential()

    # First extract all features
    for b,(images, labels, _, filenames) in enumerate(tqdm(test_loader)):
        inputs = Variable(images.cuda(), requires_grad=False)
        # Get logits
        features = feat_model(inputs)
        outputs = model.module.head.forward(features)
        U = torch.norm(features, p=1, dim=1)
        out_softmax = torch.nn.functional.softmax(outputs, dim=1)
        V = torch.norm((targets - out_softmax), p=1, dim=1)
        S = U * V / 768 / num_classes
        
        # id_preds.extend(out_softmax.argmax(dim=-1).detach().cpu().numpy())
        id_preds.extend(out_softmax.detach().cpu().numpy())
        gradnorm_preds.extend(S.detach().cpu().numpy())

        save_labels.extend(labels.detach().cpu().numpy())
        image_names.extend(filenames)
        
    id_preds = np.array(id_preds)
    gradnorm_preds = np.array(gradnorm_preds)
    save_labels = np.array(save_labels)
    
    return id_preds, gradnorm_preds, save_labels, image_names



def get_tencrop_dataload(imagenet_1k_root, imagenet_21k_root, batch_size, input_size, idx, resize_ratio=0.875, data_json_path="./splits/imagenet_ssb_splits.json"):
    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(int(input_size/resize_ratio)),
                                                torchvision.transforms.TenCrop(input_size),
                                                torchvision.transforms.Lambda(lambda crops: crops[idx]),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))])

    datasets = data.dataset_osr_test.get_imagenet_osr_test_datasets(test_transform=transform,
                                imagenet_1k_root=imagenet_1k_root,
                                imagenet_21k_root=imagenet_21k_root,
                                data_path=data_json_path)
    
    dataloaders = {}
    for k, v, in datasets.items():
        if v is not None:
            dataloaders[k] = DataLoader(v, batch_size, pin_memory=True, drop_last=False, sampler=None, num_workers=4)

    return dataloaders['test_known'], dataloaders['test_unknown']


def get_fivecrop_dataload(imagenet_1k_root, imagenet_21k_root, batch_size, input_size, idx, resize_ratio=0.875, data_json_path="./splits/imagenet_ssb_splits.json"):
    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(int(input_size/resize_ratio)),
                                                torchvision.transforms.FiveCrop(input_size),
                                                torchvision.transforms.Lambda(lambda crops: crops[idx]),
                                                # torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3),
                                                # torchvision.transforms.RandomHorizontalFlip(1.0),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))])

    datasets = data.dataset_osr_test.get_imagenet_osr_test_datasets(test_transform=transform,
                                imagenet_1k_root=imagenet_1k_root,
                                imagenet_21k_root=imagenet_21k_root,
                                data_path=data_json_path)
    
    dataloaders = {}
    for k, v, in datasets.items():
        if v is not None:
            dataloaders[k] = DataLoader(v, batch_size, pin_memory=True, drop_last=False, sampler=None, num_workers=4)

    return dataloaders['test_known'], dataloaders['test_unknown']

if __name__ == "__main__":
    
    args = utils.test_option.get_args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(100)

    iter_num_dict = {
        "10c" : 10,
        "5c" : 5,
    }

    mode = args.mode
    preds_mode = "GradNorm_RP"
    iter_num = iter_num_dict[mode]
    save_dir =args.save_dir
    save_dir = os.path.join(save_dir, mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = args.save_pth
    net = model.get_model.get_model(1000, args)
    if args.optim_name == 'fmfp' or args.optim_name == 'swa':
        net = AveragedModel(net)
    net.load_state_dict(torch.load(model_path), strict=True)
    net = net.cuda()
    
    net.module.norm = nn.Identity() 

    imagenet_1k_root = r"/datassd/Inet1K/"
    imagenet_1k_train = os.path.join(imagenet_1k_root, "train")
    imagenet_21k_root = r"/data/ImageNet-21K/"
    data_json_path = r"./splits/imagenet_ssb_splits.json"

    batch_size = args.batch_size
    input_size = args.input_size
    crop_resize_ratio = args.crop_ratio
    print(f"crop_resize_ratio: {crop_resize_ratio}")
    for idx in range(iter_num):
        if mode == "5c":
            dataloader_id, dataloader_ood = get_fivecrop_dataload(imagenet_1k_root, imagenet_21k_root, batch_size, input_size, idx, resize_ratio=crop_resize_ratio, data_json_path=data_json_path)
        elif mode == "10c":
            dataloader_id, dataloader_ood = get_tencrop_dataload(imagenet_1k_root, imagenet_21k_root, batch_size, input_size, idx, resize_ratio=crop_resize_ratio, data_json_path=data_json_path)
         
        print(len(dataloader_id.dataset))
        print(len(dataloader_ood.dataset))

        save_sub_dir = os.path.join(save_dir, str(idx))
        if not os.path.exists(save_sub_dir):
            os.makedirs(save_sub_dir)


        target = get_target(imagenet_1k_train)
        id_preds, osr_preds_id_samples, id_labels, id_image_names = test_predict_GradNorm_RP(net, dataloader_id, target)
        _, osr_preds_osr_samples, _, ood_image_names = test_predict_GradNorm_RP(net, dataloader_ood, target)

        np.save(os.path.join(save_sub_dir, "id_preds_softmax.npy"), id_preds)
        np.save(os.path.join(save_sub_dir, "id_preds_labels.npy"), id_labels)
        np.save(os.path.join(save_sub_dir, "id_preds_score.npy"), osr_preds_id_samples)
        np.save(os.path.join(save_sub_dir, "ood_preds_score.npy"), osr_preds_osr_samples)

        id_image_names_w = [name + "\n" for name in id_image_names]
        with open(os.path.join(save_sub_dir, "id_image_name.txt"), 'w') as fd:
            fd.writelines(id_image_names_w)
        ood_image_names_w = [name + "\n" for name in ood_image_names]
        with open(os.path.join(save_sub_dir, "ood_image_name.txt"), 'w') as fd:
            fd.writelines(ood_image_names_w)

        print(f"save predict {idx} Successful!")
