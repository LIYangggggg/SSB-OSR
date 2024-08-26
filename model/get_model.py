import model.classifier
import timm
from timm.models import create_model
import torch


def get_model(nb_cls, args):
    
    net = timm.create_model('deit3_base_patch16_384', checkpoint_path=args.deit_path).cuda()
    
    num_ftrs = net.head.in_features
    if args.use_cosine:
        net.head = model.classifier.Classifier(num_ftrs, nb_cls, args.cos_temp).cuda()
        if 'distilled' in args.deit_path :
            net.head_dist = model.classifier.Classifier(num_ftrs, nb_cls, args.cos_temp).cuda()

    else:
        net.head = torch.nn.Linear(num_ftrs, nb_cls).cuda()
        if 'distilled' in args.deit_path :
            net.head_dist = torch.nn.Linear(num_ftrs, nb_cls).cuda()

    return net
