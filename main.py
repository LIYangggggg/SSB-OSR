import torch.backends.cudnn
import torch.utils.tensorboard
import os
import json
import data.dataset_osr_test
import train
import utils.test_osr_ood
from torch.optim.swa_utils import AveragedModel
import model.get_model
import utils.optim as optim
import data.dataset
import utils.utils
import utils.option

# the multi GPU setting support
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import resource

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return str(port)


def main(proc_idx, args):
    gpu_id = proc_idx
    rank = int(gpu_id)
    torch.cuda.set_device(proc_idx)


    save_path = os.path.join(args.save_dir, f"{args.data_name}_{args.optim_name}-mixup_{args.mixup_weight}-crl_{args.crl_weight}")
    if rank == 0:
        writer = torch.utils.tensorboard.SummaryWriter(save_path)
        logger = utils.utils.get_logger(save_path)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    else:
        writer, logger = None, None

    # if len(args.gpu) > 1:
    dist.init_process_group('nccl', rank=rank, world_size=len(args.gpu))

    train_loader, valid_loader, _, nb_cls = data.dataset.get_loader(args.data_name, args.train_dir, args.val_dir, args.test_dir,
                                                                        args.batch_size, args.gpu)

    # we don't use the valid_loader before in the osr competition
    valid_loader_id, valid_loader_ood = data.dataset_osr_test.get_dataload(args.gpu, args.id_dir, args.ood_dir, args.batch_size)
        
    ## define model, optimizer 
    net = model.get_model.get_model(nb_cls, args)
    
    if rank == 0:
        print(net)
    if args.resume:
        if args.optim_name == 'fmfp' or args.optim_name == 'swa':
            net = AveragedModel(net)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net.pth')))
        logger.info(f"Loading checkpoints from {save_path}")
    net = net.to(rank)

    optimizer, cos_scheduler, swa_model, swa_scheduler = optim.get_optimizer_scheduler(args.optim_name,
                                                                                        net,
                                                                                        args.lr,
                                                                                        args.momentum,
                                                                                        args.weight_decay,
                                                                                        max_epoch_cos = args.epochs,
                                                                                        swa_lr = args.swa_lr)


    # multi GPU support
    net = DDP(net, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)

    # make logger
    correct_log, best_acc, best_auroc, best_aurc = train.Correctness_Log(len(train_loader.dataset)), 0, 0, 1e6

    # start Train
    for epoch in range(1, args.epochs + 2):
        train.train(train_loader, net, optimizer, epoch, correct_log, logger, writer, args)

        if args.optim_name in ['swa', 'fmfp'] :
            if epoch > args.swa_epoch_start:
                if len(args.gpu) > 1:
                    swa_model.update_parameters(net.module)
                else:
                    swa_model.update_parameters(net)
                swa_scheduler.step()
            else:
                cos_scheduler.step()
        else:
            cos_scheduler.step()

        # validation
        if epoch > args.swa_epoch_start and args.optim_name in ['swa', 'fmfp'] :
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device='cuda')
            net_val = swa_model.cuda()
        else : 
            net_val = net
            
        # evaluate 
        if rank == 0:    
            res = utils.test_osr_ood.get_osr_ood_metric(net_val, valid_loader_id, valid_loader_ood) # change here for an osr competition validation
            log = [key + ': {:.3f}'.format(res[key]) for key in res]
            msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            logger.info(msg)

            for key in res :
                writer.add_scalar('./Val/' + key, res[key], epoch)

            if res['ACC'] > best_acc :
                acc = res['ACC']
                msg = f'Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
                logger.info(msg)
                best_acc = acc
                torch.save(net_val.state_dict(), os.path.join(save_path, 'best_acc_net.pth'))
                
        synchronize()

if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    args = utils.option.get_args_parser()
    torch.backends.cudnn.benchmark = True

    save_path = os.path.join(args.save_dir, f"{args.data_name}_{args.optim_name}-mixup_{args.mixup_weight}-crl_{args.crl_weight}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))

    ## fix avilabel port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = find_free_port()
    if len(args.gpu) == 1:
        main(0, args)
    else:
        mp.spawn(main, nprocs=len(args.gpu), args=(args,))




