#!/usr/bin/env python3
# Train Q-SNN ResNet-20 on TinyImageNet using pass-based quantization (1w-2u)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import time, datetime
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from collections import OrderedDict
from utils import data_loaders
from utils import common
from utils.functions import split_weights

from models.pure_resnet import resnet_20
from chop.passes.module.transforms import quantize_module_transform_pass
from models.layers import LIFSpikeQ


parser = argparse.ArgumentParser("TinyImageNet ResNet-20 Q-SNN (1w-2u) Quant + Train")

parser.add_argument('--arch', type=str, default='resnet_20', help='architecture')
parser.add_argument('--job_dir', type=str, default='/workspace/QP-SNN-Quantization-pass/output_resnet_quan/Q-SNN/TinyImageNet/QSNN_1w2/', help='path for saving models')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--resume', action='store_true', help='whether continue training from the same directory')
parser.add_argument('--gpu', type=str, default='0', help='Select gpu to use')
parser.add_argument('--dataset', default='TinyImageNet', type=str, help='dataset name', choices=['TinyImageNet'])
parser.add_argument('-j','--workers', default=8, type=int, metavar='N', help='number of data loading workers')

args = parser.parse_args()
print_freq = 50
common.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = common.get_logger(os.path.join(args.job_dir, 'ResNet_TinyImageNet_QSNN_1w2_'+now+'.log'))

if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)


def create_quantized_model_via_pass(num_classes):
    logger.info('==> Creating ResNet-20 via Q-SNN pass (1w-2u)..')
    model = resnet_20(compress_rate=[0.0]*12, num_classes=num_classes)
    for p in model.parameters():
        p.requires_grad = True

    pass_args = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q": LIFSpikeQ},
        # first conv
        r"conv1_s\.layer\.module$": {
            "config": {"name": "apot", "num_bits": 2, "base_k": 2}
        },
        # hidden convs
        r"layer\d+\.\d+\.conv[12]_s\.layer\.module$": {
            "config": {"name": "qsnn"}
        },
        # final fc
        r"fc\.module$": {
            "config": {"name": "apot", "num_bits": 2, "base_k": 2}
        },
        # voltage quantization for LIF neurons (T=2 in resnet_20)
        r"relu$": {"config": {"name": "lifspike_q", "num_bits": 2, "thresh": 1.0, "tau": 0.5}},
        r"layer\d+\.\d+\.relu[12]$": {"config": {"name": "lifspike_q", "num_bits": 2, "thresh": 1.0, "tau": 0.5}},
    }

    model, _ = quantize_module_transform_pass(model, pass_args)
    logger.info('==> Q-SNN pass done (1w-8u, T=2, tau=0.5, thresh=1.0)')
    return model


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = common.AverageMeter('Time', ':6.3f')
    data_time = common.AverageMeter('Data', ':6.3f')
    losses = common.AverageMeter('Loss', ':.4e')
    top1 = common.AverageMeter('Acc@1', ':6.2f')

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        logits = model(images)
        out = logits.mean(1)
        loss = criterion(out, target)

        prec1 = common.accuracy(out, target, topk=(1,))[0]
        n = images.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Epoch[{0}]({1}/{2}): Loss {loss.avg:.4f} Prec@1(1) {top1.avg:.2f}'
                        .format(epoch, i, num_iter, loss=losses, top1=top1))

    scheduler.step()
    return losses.avg, top1.avg


def validate(epoch, val_loader, model, criterion, args):
    batch_time = common.AverageMeter('Time', ':6.3f')
    losses = common.AverageMeter('Loss', ':.4e')
    top1 = common.AverageMeter('Acc@1', ':6.2f')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            logits = model(images)
            out = logits.mean(1)
            loss = criterion(out, target)

            pred1 = common.accuracy(out, target, topk=(1, ))[0]
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)

            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg


def main():
    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)

    # TinyImageNet dataset
    trainset, testset = data_loaders.build_tiny_imagenet()
    CLASSES = 200
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    logger.info('==> Building model..')
    model = create_quantized_model_via_pass(num_classes=CLASSES)
    model.to(device)
    logger.info(model)

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    criterion = nn.CrossEntropyLoss().to(device)

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    # Per paper for CIFAR-10: SGD + cosine, lr=0.1, batch=256 (WD not specified; use 1e-4)
    optimizer = torch.optim.SGD(
        [{'params': other_parameters},
         {'params': weight_parameters, 'weight_decay': 1e-4}],
        lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    start_epoch = 0
    best_top1_acc= 0

    if args.resume:
        checkpoint_dir = os.path.join(args.job_dir, 'checkpoint.pth.tar')
        logger.info('loading checkpoint {} ..........'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("loaded checkpoint {} epoch = {}".format(checkpoint_dir, checkpoint['epoch']))
        for epoch in range(start_epoch):
            scheduler.step()
    else:
        logger.info('training from scratch')

    epoch = start_epoch
    while epoch < args.epochs:
        train_obj, train_top1_acc = train(epoch,  train_loader, model, criterion, optimizer, scheduler)
        valid_obj, valid_top1_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        common.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.job_dir)

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))


if __name__ == '__main__':
    main()


