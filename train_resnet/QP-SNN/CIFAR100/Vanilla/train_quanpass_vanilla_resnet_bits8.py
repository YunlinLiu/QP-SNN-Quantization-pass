import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
# from thop import profile, clever_format
import numpy as np
 

# ===== Import pure_resnet and quantization pass =====
from models.quant_vgg import vgg_16_bn
from models.pure_resnet import resnet_20
from mase.src.chop.passes.module.transforms import quantize_module_transform_pass


parser = argparse.ArgumentParser("CIFAR-100 ResNet-20 Vanilla Quantization (8-bit)")

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_20',
    help='architecture')

parser.add_argument(
    '--job_dir',
    type=str,
    default='./output_resnet_quan/CIFAR100/Vanilla_q8/',  # Output to Vanilla_q8 folder
    help='path for saving trained models')

parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='batch size')

parser.add_argument(
    '--epochs',
    type=int,
    default=300,
    help='num of training epochs')

parser.add_argument(
    '--lr',
    type=float,
    default=0.1,
    help='init learning rate')

parser.add_argument(
    '--resume',
    action='store_true',
    help='whether continue training from the same directory')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

parser.add_argument(
    '--dataset',
    default='CIFAR100',
    type=str,
    help='dataset name',
    choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])

parser.add_argument(
    '-j',
    '--workers',
    default=8,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 16)')

parser.add_argument(
    '-bit',
    default=8,
    type=int,
    metavar='N',
    help='bitwidth of weight')

args = parser.parse_args()
print_freq = 50
common.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = common.get_logger(os.path.join(args.job_dir, 'ResNet_CIFAR100_logger'+now+'.log'))

if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)


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
            logger.info(
                'Epoch[{0}]({1}/{2}): Loss {loss.avg:.4f} Prec@1(1) {top1.avg:.2f}'
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

    # load training data
    if args.dataset == 'CIFAR10':
        trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=True, download=True)
        CLASSES = 10
    elif args.dataset == 'CIFAR100':
        trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=False, download=True)
        CLASSES = 100
    elif args.dataset == 'ImageNet':
        trainset, testset = data_loaders.build_imagenet()
        CLASSES = 1000
    elif args.dataset == 'DVSCIFAR10':
        trainset, testset = data_loaders.build_dvscifar()
        CLASSES = 10
    elif args.dataset == 'TinyImageNet':
        trainset, testset = data_loaders.build_tiny_imagenet()
        CLASSES = 200
    elif args.dataset == 'DVS128':
        trainset, testset = data_loaders.build_dvs128(T=args.time)
        CLASSES = 11
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # load model
    logger.info('==> Building model..')
    logger.info('=== Bit width===:'+str(args.bit))

    # 1) Create pure model
    model = eval(args.arch)(compress_rate=[0.]*12, num_classes=CLASSES)

    # 2) Set requires_grad for QAT
    for param in model.parameters():
        param.requires_grad = True

    # 3) Apply quantization pass (Vanilla)
    quan_pass_args = {
        "by": "regex_name",
        # Quantize Conv2d layers inside tdLayer (conv1_s, conv2_s), exclude the first conv1
        r"layer\d+\.\d+\.conv[12]_s\.layer\.module$": {
            "config": {
                "name": "vanilla",
                "num_bits": args.bit,
            }
        },
    }
    model, _ = quantize_module_transform_pass(model, quan_pass_args)
    logger.info('==> Applied Vanilla quantization pass with %d bits' % args.bit)

    # 4) Move to device
    model.to(device)
    logger.info(model)

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.SGD(
        [{'params': other_parameters},
         {'params': weight_parameters, 'weight_decay': 1e-5}],
        lr=args.lr,
        momentum=0.9)

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


