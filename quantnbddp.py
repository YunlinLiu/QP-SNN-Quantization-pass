# quant_ddp.py
import os
import time, datetime
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils import data_loaders
from utils import common
from models.quant_vgg import vgg_16_bn
from models.quant_resnet_cifar import resnet_20


def parse_args():
    p = argparse.ArgumentParser("cifar ddp")
    p.add_argument('--arch', type=str, default='vgg_16_bn', help='vgg_16_bn or resnet_20')
    p.add_argument('--job_dir', type=str, default='./log/', help='output dir')
    p.add_argument('--batch_size', type=int, default=256, help='PER-GPU batch size')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dataset', type=str, default='CIFAR10',
                   choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])
    p.add_argument('-j', '--workers', type=int, default=8)
    p.add_argument('-bit', type=int, default=8, metavar='N', help='bitwidth of weight')

    # DDP/run-time
    p.add_argument('--dist_backend', default='nccl')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def init_distributed():
    """
    Initialize torch.distributed from torchrun-env.
    Returns (is_distributed, rank, local_rank, world_size, device)
    """
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
        return True, rank, local_rank, world_size, device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return False, 0, 0, 1, device


def is_main(rank): return rank == 0


def build_dataloaders(args, world_size, rank):
    # datasets
    if args.dataset == 'CIFAR10':
        trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=True, download=False)
        classes = 10
    elif args.dataset == 'CIFAR100':
        trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=False, download=False)
        classes = 100
    elif args.dataset == 'ImageNet':
        trainset, testset = data_loaders.build_imagenet()
        classes = 1000
    elif args.dataset == 'TinyImageNet':
        trainset, testset = data_loaders.build_tiny_imagenet()
        classes = 200
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    # samplers
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) \
        if world_size > 1 else None
    val_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) \
        if world_size > 1 else None

    # loaders
    pin = True
    pf = 4
    persistent = args.workers > 0
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=pf if args.workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, args.workers // 2),
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=pf if args.workers > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler, val_sampler, classes


def maybe_convert_state_dict_for_model(state_dict, model_is_ddp):
    """Add/remove 'module.' prefix to match current model wrapper."""
    has_module = all(k.startswith('module.') for k in state_dict.keys())
    new_sd = OrderedDict()
    if model_is_ddp and not has_module:
        for k, v in state_dict.items():
            new_sd['module.' + k] = v
        return new_sd
    if (not model_is_ddp) and has_module:
        for k, v in state_dict.items():
            new_sd[k.replace('module.', '', 1)] = v
        return new_sd
    return state_dict


def save_ckpt(model, optimizer, epoch, best_acc, job_dir, rank):
    if not is_main(rank):
        return
    os.makedirs(job_dir, exist_ok=True)
    # 保存为不带 'module.' 的权重，通吃单卡/DP/DDP
    raw_sd = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
    torch.save({
        'epoch': epoch,
        'state_dict': raw_sd,
        'best_top1_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, os.path.join(job_dir, 'checkpoint.pth.tar'))


def train_one_epoch(epoch, loader, sampler, model, criterion, optimizer, scheduler, device, logger, world_size):
    if sampler is not None:
        sampler.set_epoch(epoch)

    model.train()
    batch_time = common.AverageMeter('Time', ':6.3f')
    data_time = common.AverageMeter('Data', ':6.3f')
    losses = common.AverageMeter('Loss', ':.4e')
    top1 = common.AverageMeter('Acc@1', ':6.2f')

    end = time.time()
    # 打印频率（确保 >=1）
    print_freq = max( (256 * 50) // max(1, loader.batch_size), 1 )

    # log lr
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    for i, (images, target) in enumerate(loader):
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(images)
        out = logits.mean(1)
        loss = criterion(out, target)

        # metrics (local)
        with torch.no_grad():
            prec1 = common.accuracy(out, target, topk=(1,))[0].item()
        n = images.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1, n)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and logger is not None:
            logger.info('Epoch[{0}]({1}/{2}): Loss {loss.avg:.4f} Prec@1(1) {top1.avg:.2f}'
                        .format(epoch, i, len(loader), loss=losses, top1=top1))

    scheduler.step()

    # 统计跨进程平均（可选）
    if world_size > 1:
        t = torch.tensor([losses.sum, losses.count, top1.sum, top1.count],
                         dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_avg = (t[0] / t[1]).item()
        acc1_avg = (t[2] / t[3]).item()
    else:
        loss_avg, acc1_avg = losses.avg, top1.avg

    return loss_avg, acc1_avg


@torch.no_grad()
def validate(epoch, loader, sampler, model, criterion, device, logger, world_size):
    if sampler is not None and hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(0)  # 不打乱

    model.eval()
    losses_sum = torch.tensor(0.0, device=device)
    losses_cnt = torch.tensor(0.0, device=device)
    correct_sum = torch.tensor(0.0, device=device)
    samples_cnt = torch.tensor(0.0, device=device)

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(images)
        out = logits.mean(1)
        loss = criterion(out, target)

        pred1 = common.accuracy(out, target, topk=(1,))[0].item()
        n = images.size(0)

        losses_sum += loss.item() * n
        losses_cnt += n
        correct_sum += pred1 * n
        samples_cnt += n

    if world_size > 1:
        dist.all_reduce(losses_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(losses_cnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_cnt, op=dist.ReduceOp.SUM)

    loss_avg = (losses_sum / losses_cnt).item()
    top1_avg = (correct_sum / samples_cnt).item()

    if logger is not None:
        logger.info(' * Acc@1 {:.3f}'.format(top1_avg))
    return loss_avg, top1_avg


def main():
    args = parse_args()
    cudnn.benchmark = True
    cudnn.enabled = True

    # DDP init
    dist_on, rank, local_rank, world_size, device = init_distributed()

    # logger（只在 rank0 记录）
    os.makedirs(args.job_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger = common.get_logger(os.path.join(args.job_dir, f'logger{now}.log')) if is_main(rank) else None
    if is_main(rank):
        common.record_config(args)
        print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}")

    # data
    train_loader, val_loader, train_sampler, val_sampler, classes = build_dataloaders(args, world_size, rank)

    # model
    if is_main(rank):
        (logger or print).info('==> Building model..')
        (logger or print).info('=== Bit width===:' + str(args.bit))
    model = eval(args.arch)(compress_rate=[0.]*100, num_bits=args.bit, num_classes=classes)
    model.to(device)

    if dist_on:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # loss/opt/sched
    criterion = nn.CrossEntropyLoss().to(device)
    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
    other_parameters = [p for p in all_parameters if p not in weight_parameters]
    optimizer = torch.optim.Adam(
        [{'params': other_parameters},
         {'params': weight_parameters, 'weight_decay': 1e-5}],
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    # resume
    start_epoch = 0
    best_top1 = 0.0
    ckpt_path = os.path.join(args.job_dir, 'checkpoint.pth.tar')
    if args.resume and os.path.isfile(ckpt_path):
        map_loc = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if dist_on else None
        if is_main(rank):
            (logger or print).info(f'loading checkpoint {ckpt_path} ..........')
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best_top1 = float(ckpt.get('best_top1_acc', 0.0))
        state_dict = ckpt['state_dict']
        state_dict = maybe_convert_state_dict_for_model(state_dict, model_is_ddp=isinstance(model, DDP))
        model.load_state_dict(state_dict, strict=True)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        # 追赶学习率
        for _ in range(start_epoch):
            scheduler.step()
        if is_main(rank):
            (logger or print).info(f'loaded checkpoint epoch={ckpt.get("epoch", "?")} best={best_top1:.3f}')
    elif is_main(rank):
        (logger or print).info('training from scratch')

    # train loop
    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_acc = train_one_epoch(epoch, train_loader, train_sampler,
                                          model, criterion, optimizer, scheduler,
                                          device, logger, world_size)
        va_loss, va_acc = validate(epoch, val_loader, val_sampler,
                                   model, criterion, device, logger, world_size)

        if va_acc > best_top1:
            best_top1 = va_acc
        save_ckpt(model, optimizer, epoch, best_top1, args.job_dir, rank)
        if is_main(rank) and logger is not None:
            logger.info("=>Best accuracy {:.3f}".format(best_top1))

    if dist_on:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
