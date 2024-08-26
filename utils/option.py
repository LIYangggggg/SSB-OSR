import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Failure prediction framework',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--epochs', default=200, type=int, help='Total number of training epochs ')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    
    ## optimizer
    parser.add_argument('--lr', default=0.1, type=float, help='Max learning rate for cosine learning rate scheduler')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')


    ## nb of run + print freq
    parser.add_argument('--nb-run', default=3, type=int, help='Run n times, in order to compute std')

    ## dataset setting
    parser.add_argument('--nb-worker', default=4, type=int, help='Nb of workers')
    parser.add_argument('--mixup-beta', default=10.0, type=float, help='beta used in the mixup data aug')

    parser.add_argument('--optim-name', default='baseline', type=str, choices=['baseline', 'sam', 'swa', 'fmfp'],
                        help='Supported methods for optimization process')

    parser.add_argument('--save-dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--resume', action='store_true', default=False, help='whether resume training')

    # the new network setting
    parser.add_argument('--model', default='deit_base_patch16_LS', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.15, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--input-size', default=384, type=int, help='images input size')
    parser.add_argument('--distillation', default=False, action='store_true', help='whether use head distillation')
    parser.add_argument('--finetune', default='', help='finetune for checkpoint')
    
    ## cosine classifier
    parser.add_argument('--use-cosine', action='store_true', default=False, help='whether use cosine classifier ')
    parser.add_argument('--cos-temp', type=int, default=8, help='temperature for scaling cosine similarity')

    ''' the original getting model configs, is now muted
    parser.add_argument('--train-size', default=224, type=int, help='train size')
    
    ## Energy parameters
    parser.add_argument('--m-in', default=-25., type=float, help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--energy-weight', default=0.1, type=float, help='energy_weight')
    
    ## Model + optim method + data aug + loss + post-hoc
    parser.add_argument('--model-name', default='resnet18', type=str,choices = ['resnet18', 'resnet32', 'resnet50', 'densenet', 'wrn', 'vgg', 'vgg19bn', 'deit'],
                        help='Models name to use')
    parser.add_argument('--resume-path', type=str, help='resume path')
    parser.add_argument('--deit-path', default = '/home/liyuting/ICLR/SURE-main/deit_base_patch16_224-b5f2ef4d.pth', type=str, help='Official DeiT checkpoints')

    ## num register
    parser.add_argument('--num-register', type=int, default=4, help='number of register')

    ## fine-tuning
    parser.add_argument('--fine-tune-epochs', default=20, type=int, help='Total number of fine-tuning ')
    parser.add_argument('--fine-tune-lr', default=0.01, type=float,
                        help='Max learning rate for cosine learning rate scheduler')
    parser.add_argument('--reweighting-type', default=None, type=str, choices=['exp', 'threshold', 'power', 'linear'])
    parser.add_argument('--alpha', default=0.5, type=float, help='When you set re-weighting type to [threshold], you can set the threshold by changing alpha')
    parser.add_argument('--p', default=2, type=int, help='When you set re-weighting type to [power], you can set the power by changing p')
    parser.add_argument('--t', default=1.0, type=float, help='When you set re-weighting type to [exp], you can set the temperature by changing t')
    '''
    
    parser.add_argument('--crl-weight', default=0.0, type=float, help='CRL loss weight')
    parser.add_argument('--mixup-weight', default=0.0, type=float, help='Mixup loss weight')
    parser.add_argument('--gpu', default=[2,3,4,5], type=int, nargs='+', help='GPU ids to use')


    ## SWA parameters
    parser.add_argument('--swa-lr', default=0.05, type=float, help='swa learning rate')
    parser.add_argument('--swa-epoch-start', default=120, type=int, help='swa start epoch')
    
    ## dataset setting
    subparsers = parser.add_subparsers(title="dataset setting", dest="subcommand")
    
    Cifar10 = subparsers.add_parser("Cifar10",
                                    description='Dataset parser for training on Cifar10',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Dataset parser for training on Cifar10")
    Cifar10.add_argument('--data-name', default='cifar10', type=str, help='Dataset name')
    Cifar10.add_argument("--train-dir", type=str, default='./data/CIFAR10/train',
                         help="Cifar10 train directory")
    Cifar10.add_argument("--val-dir", type=str, default='./data/CIFAR10/val', help="Cifar10 val directory")
    Cifar10.add_argument("--test-dir", type=str, default='./data/CIFAR10/test', help="Cifar10 val directory")
    Cifar10.add_argument("--corruption-dir", type=str, default='./data', help="Cifar10 val directory")
    Cifar10.add_argument("--nb-cls", type=int, default=10, help="number of classes in Cifar10")
    Cifar10.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in Cifar10")

    Cifar100 = subparsers.add_parser("Cifar100",
                                     description='Dataset parser for training on Cifar100',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     help="Dataset parser for training on Cifar100")
    Cifar100.add_argument('--data-name', default='cifar100', type=str, help='Dataset name')
    Cifar100.add_argument("--train-dir", type=str, default='./data/CIFAR100/train', help="Cifar100 train directory")
    Cifar100.add_argument("--val-dir", type=str, default='./data/CIFAR100/val', help="Cifar100 val directory")
    Cifar100.add_argument("--test-dir", type=str, default='./data/CIFAR100/test', help="Cifar100 val directory")
    Cifar100.add_argument("--nb-cls", type=int, default=100, help="number of classes in Cifar100")
    Cifar100.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in Cifar100")


    TinyImgNet = subparsers.add_parser("TinyImgNet",
                                       description='Dataset parser for training on TinyImgNet',
                                       add_help=True,
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="Dataset parser for training on TinyImgNet")
    TinyImgNet.add_argument('--data-name', default='TinyImgNet', type=str, help='Dataset name')
    TinyImgNet.add_argument("--train-dir", type=str, default='./data/tinyImageNet/train',
                            help="TinyImgNet train directory")
    TinyImgNet.add_argument("--val-dir", type=str, default='./data/tinyImageNet/val', help="TinyImgNet val directory")
    TinyImgNet.add_argument("--test-dir", type=str, default='./data/tinyImageNet/test', help="TinyImgNet val directory")
    TinyImgNet.add_argument("--nb-cls", type=int, default=200, help="number of classes in TinyImgNet")
    TinyImgNet.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in TinyImgNet")

    ImgNet1k = subparsers.add_parser("ImgNet1k",
                                       description='Dataset parser for training on ImgNet1k',
                                       add_help=True,
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="Dataset parser for training on TinyImgNet")
    ImgNet1k.add_argument('--data-name', default='imagenet', type=str, help='Dataset name')
    ImgNet1k.add_argument("--train-dir", type=str, default='/datassd/Inet1K/train', help="ImgNet1K train directory")
    ImgNet1k.add_argument("--val-dir", type=str, default='/datassd/Inet1K/val', help="ImgNet1K val directory")
    ImgNet1k.add_argument("--test-dir", type=str, default='/datassd/Inet1K/val', help="ImgNet1K test directory")
    # add a path leading to OOD position for OSR competition
    ImgNet1k.add_argument("--id-dir", type=str, default='/datassd/Inet1K', help="ID val directory")
    ImgNet1k.add_argument("--ood-dir", type=str, default='/data/ImageNet-21K', help="OOD val directory")
    ImgNet1k.add_argument("--nb-cls", type=int, default=1000, help="number of classes in ImgNet1K")
    ImgNet1k.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in ImgNet1K")

    return parser.parse_args()
