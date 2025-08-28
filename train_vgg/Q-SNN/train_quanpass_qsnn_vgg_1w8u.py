#!/usr/bin/env python3
# Train Q-SNN VGG16 on TinyImageNet using pass-based quantization (1w-8u)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from pathlib import Path
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

from chop.passes.module.transforms import quantize_module_transform_pass

from utils import data_loaders
from utils import common
from utils.functions import split_weights

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
from models.pure_vgg import vgg_16_bn
from models.layers import LIFSpikeQ


parser = argparse.ArgumentParser("TinyImageNet VGG-16 Q-SNN (1w-8u) Quantization + Train")

parser.add_argument('--arch', type=str, default='vgg_16_bn', help='architecture')
parser.add_argument('--job_dir', type=str, default='./output_vgg_quan/Q-SNN/TinyImageNet/QSNN_1w8u/', help='path for saving models')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='num of epochs')
parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint in job_dir')
parser.add_argument('--gpu', type=str, default='0', help='Select gpu to use')
parser.add_argument('--dataset', default='TinyImageNet', type=str, choices=['TinyImageNet'], help='dataset name')

args = parser.parse_args()
print_freq = (256*50)//args.batch_size

common.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = common.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))

if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)


def create_quantized_model_via_pass(compress_rate, num_classes):
    logger.info('==> Creating model via Q-SNN pass (1w-8u)..')

    vgg = vgg_16_bn(compress_rate=compress_rate, num_classes=num_classes)
    vgg.T = 4  # T=4 for TinyImageNet as per paper
    for param in vgg.parameters():
        param.requires_grad = True

    pass_args = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q": LIFSpikeQ},
        r"features\.convbn0\.layer\.module": {
            "config": {"name": "apot", "num_bits": 8, "base_k": 2}
        },
        r"features\.convbn(?!0\b)\d+\.layer\.module": {
            "config": {"name": "qsnn"}
        },
        r"classifier\.linear1\.module": {
            "config": {"name": "apot", "num_bits": 8, "base_k": 2}
        },
        r"features\.relu\d+": {
            "config": {"name": "lifspike_q", "num_bits": 8, "thresh": 1.0, "tau": 0.5}
        },
    }

    mg, _ = quantize_module_transform_pass(vgg, pass_args)
    logger.info('==> Q-SNN pass done (1w-8u, T=4, tau=0.5, thresh=1.0)')
    return mg


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
    cudnn.enabled = True
    logger.info("args = %s", args)

    # Load TinyImageNet dataset
    trainset, testset = data_loaders.build_tiny_imagenet()
    CLASSES = 200  # TinyImageNet has 200 classes
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    logger.info('==> Building model..')
    model = create_quantized_model_via_pass(compress_rate=[0.]*100, num_classes=CLASSES)
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

    # Adam optimizer for TinyImageNet per paper; batch=256; cosine LR schedule; initial lr=0.1
    optimizer = torch.optim.Adam(
        [{'params': other_parameters},
         {'params': weight_parameters, 'weight_decay': 1e-4}],
        lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    start_epoch = 0
    best_top1_acc = 0

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
        train_obj, train_top1_acc = train(epoch, train_loader, model, criterion, optimizer, scheduler)
        valid_obj, valid_top1_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        common.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.job_dir)

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))


if __name__ == '__main__':
    main()


