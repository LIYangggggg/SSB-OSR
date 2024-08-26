import os
import numpy as np
import json
from torchvision import transforms
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader, make_dataset, IMG_EXTENSIONS

IMG_EXTENSIONS += ('.JPEG',)


"""
OoD evaluation utils from: https://github.com/deeplearning-wisc/react/blob/master/util/metrics.py
NOTE: '1' is considered ID/Known, and '0' is considered OoD/Unkown
"""

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_class_splits(data_path: str) -> dict:
    """
    Load known/unknown class splits of a dataset.

    Arguments:
    json_path -- dataset json path.

    Returns: Dictionary with known classes and unknown classes
    """

    with open(data_path, 'r') as f:
        class_splits = json.load(f)

    return class_splits


class CustomImageFolder(torchvision.datasets.ImageFolder):

    """
    Base ImageFolder
    """

    def __init__(self, root, transform, dataset_name, data_path):

        self.root = root
        
        if dataset_name == 'imagenet_21k_easy':
            class_splits = load_class_splits(data_path)['unknown_classes']['Easy']
            class_name_to_index = {name: int(ind) + 1000 for ind, name in enumerate(class_splits)}      # Offset class indices by ImageNet1K clases
        elif dataset_name == 'imagenet_21k_hard':
            class_splits = load_class_splits(data_path)['unknown_classes']['Hard']
            class_name_to_index = {name: int(ind) + 1000 for ind, name in enumerate(class_splits)}      # Offset class indices by ImageNet1K clases
        elif dataset_name == 'imagenet_1k':
            class_splits = load_class_splits(data_path)['known_classes']
            class_name_to_index = {name: int(ind) for ind, name in enumerate(class_splits)}
        else:
            raise ValueError

        samples = make_dataset(root, 
                               class_name_to_index, 
                               extensions=IMG_EXTENSIONS, 
                               is_valid_file=None)

        self.imgs = samples
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = None            
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.class_to_idx = class_name_to_index

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        path, _ = self.samples[uq_idx]
        filename = os.path.basename(path)

        return img, label, uq_idx, filename


def get_imagenet_osr_test_datasets(test_transform,
                              imagenet_1k_root="../imagenet_1k_root", 
                              imagenet_21k_root="../imagenet_21k_root",
                              data_path="./splits/imagenet_ssb_splits.json"):

    """
    Create PyTorch Datasets for ImageNet (Easy unknown classes from ImageNet-21K). 

    Loads datasets for open-set recognition

    Arguments: 
    osr_split -- Unused, always the ImageNet 1K classes
    train_transform, test_transform -- Torchvision transforms
    
    Returns:
    all_datasets -- dict containing, 
        test_known: Test images from known classes
        test_unknown: Test images from unknown classes
    """

    test_dataset_known = CustomImageFolder(
            root=os.path.join(imagenet_1k_root, 'val'),
            transform=test_transform,
            dataset_name='imagenet_1k',
            data_path=data_path
        )

    test_dataset_unknown_easy = CustomImageFolder(
            root=os.path.join(imagenet_21k_root, 'imagenet21k_val'),
            transform=test_transform,
            dataset_name='imagenet_21k_easy',
            data_path=data_path
        )

    test_dataset_unknown_hard = CustomImageFolder(
            root=os.path.join(imagenet_21k_root, 'imagenet21k_val'),
            transform=test_transform,
            dataset_name='imagenet_21k_hard',
            data_path=data_path
        )

    all_datasets = {
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown_easy + test_dataset_unknown_hard
    }

    return all_datasets


def get_imagenet_standard_transform(image_size=224, crop_pct=0.875,
                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):

    test_transform_standard = transforms.Compose([
        transforms.Resize(int(image_size / crop_pct), interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ])

    return test_transform_standard



def get_dataload(gpu, imagenet_1k_root, imagenet_21k_root, batch_size, data_json_path="./splits/imagenet_ssb_splits.json"):
    #transform = get_imagenet_standard_transform()
    transform = get_deit_test_transform()
    datasets = get_imagenet_osr_test_datasets(test_transform=transform, 
                                imagenet_1k_root=imagenet_1k_root,
                                imagenet_21k_root=imagenet_21k_root,
                                data_path=data_json_path)
    
    dataloaders = {}
    for k, v, in datasets.items():
        if v is not None:
            dataloaders[k] = DataLoader(v, batch_size // len(gpu), pin_memory=True, drop_last=False, sampler=None, num_workers=4)

    return dataloaders['test_known'], dataloaders['test_unknown'], 

# this is from the original deit test tranform
# all variables are given with default setting of the original
def get_deit_test_transform(input_size=384, eval_crop_ratio=0.875):
    resize_im = input_size > 32

    t = []
    if resize_im:
        size = int(input_size / eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 384 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


if __name__ == "__main__":
    imagenet_1k_root = r"/datassd/Inet1K/"
    imagenet_21k_root = r"/data/ImageNet-21K/"
    data_json_path = r"./splits/imagenet_ssb_splits.json"
    transform = get_imagenet_standard_transform()
    datasets = get_imagenet_osr_test_datasets(test_transform=transform, 
                                imagenet_1k_root=imagenet_1k_root,
                                imagenet_21k_root=imagenet_21k_root,
                                data_path=data_json_path)
    
    for key, val in datasets.items():
        print(f"key: {key};")
        print(f"val: {val};")
        print(f"length: {len(val)}")

