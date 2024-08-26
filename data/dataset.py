import os
import numpy as np
import random
import uuid
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision.datasets import ImageFolder
from data.sampler import InfiniteSampler
from timm.data import create_transform


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2 ** 32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, is_train=False, dataset_type='normal'):
        super(CustomImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.is_train = is_train

    def __getitem__(self, index):
        img, label = super(CustomImageFolder, self).__getitem__(index)
        return img, label, index


def TrainDataLoader(img_dir, transform_train, batch_size, gpu, is_train=True, dataset_type='normal'):
    train_set = CustomImageFolder(img_dir, transform_train, is_train=is_train, dataset_type=dataset_type)
    sampler = InfiniteSampler(len(train_set), shuffle=True)

    batch_sampler = BatchSampler(sampler, batch_size // len(gpu), drop_last=True)
    dataloader_kwargs = {"num_workers": 4, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
    train_loader = DataLoader(train_set, **dataloader_kwargs)
    return train_loader

# test data loader
def TestDataLoader(img_dir, transform_test, batch_size, gpu):
    test_set = CustomImageFolder(img_dir, transform_test)
    test_loader = DataLoader(test_set, batch_size // len(gpu), pin_memory=True, num_workers=4, drop_last=False, sampler=None)

    return test_loader

def get_loader(dataset, train_dir, val_dir, test_dir, batch_size, gpu):


    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ### TODO: !!should be optimized!!
    '''
    transform_train = torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize(256),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(norm_mean, norm_std)])
    '''
    transform_train = get_deit_train_transform()
    # transformation of the test set
    ### TODO: Resolution is absolutely very important to get a good performance
    transform_test = torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize(256),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(norm_mean, norm_std)])
    if dataset == 'cifar100':
        nb_cls = 100

    elif dataset == 'imagenet':
        nb_cls = 1000


    train_loader = TrainDataLoader(train_dir, transform_train, batch_size, gpu, is_train=True, dataset_type=dataset)
    val_loader = TestDataLoader(val_dir, transform_test, batch_size, gpu)
    test_loader = TestDataLoader(test_dir, transform_test, batch_size, gpu)

    return train_loader, val_loader, test_loader, nb_cls

def get_deit_train_transform():
    transform = create_transform(
        input_size=384,
        is_training=True,
        color_jitter=0.3,
        auto_augment='rand-m9-mstd0.50-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    return transform
