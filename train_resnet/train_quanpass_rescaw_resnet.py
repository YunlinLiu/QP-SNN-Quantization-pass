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
import matplotlib.pyplot as plt

# ===== Main modification 1: Import pure_resnet instead of quant_resnet_cifar =====
from models.quant_vgg import vgg_16_bn
from models.pure_resnet import resnet_20
# ===== Add quantization pass imports =====
from mase.src.chop.passes.module.transforms import quantize_module_transform_pass


parser = argparse.ArgumentParser("CIFAR-100 ResNet-20")   # Argument parser

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_20',
    help='architecture')

parser.add_argument(
    '--job_dir',
    type=str,
    default='./log_pass/',  # Changed to log_pass to avoid confusion
    help='path for saving trained models')

parser.add_argument(
    '--batch_size',
    type=int,
    default=256,    # ResNet typically uses smaller batch size
    help='batch size')

parser.add_argument(
    '--epochs',
    type=int,
    default=150,
    help='num of training epochs')

parser.add_argument(
    '--lr',
    type=float,
    default=1e-3,
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
    default=8,    # Reduced number of workers to adapt to ResNet training
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
# print_freq = (256*50)//args.batch_size # Determine how often to print logs during training, if batch_size=256, print every 50 batches
print_freq = 50
common.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = common.get_logger(os.path.join(args.job_dir, 'ResNet_CIFAR100_logger'+now+'.log'))

if not os.path.isdir(args.job_dir): # If directory doesn't exist, recursively create all necessary parent directories
    os.makedirs(args.job_dir)

# use for loading pretrain model
# if len(args.gpu)>1:
#     name_base='module.' #在PyTorch中，当使用nn.DataParallel进行多GPU训练时，模型参数名会自动添加module.前缀。
# else:
#     name_base=''

def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = common.AverageMeter('Time', ':6.3f')
    data_time = common.AverageMeter('Data', ':6.3f')
    losses = common.AverageMeter('Loss', ':.4e')
    top1 = common.AverageMeter('Acc@1', ':6.2f')

    model.train()   # 设置模型为训练模式
    end = time.time()   # 记录开始时间

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))    # Get current learning rate and log it

    num_iter = len(train_loader)    # Total number of batches
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end) # Record data loading time
        images = images.to(device)
        target = target.to(device)

        # compute outputy
        logits = model(images)
        out = logits.mean(1)     # SNN specific: Time dimension averaging, SNN output shape is usually [batch_size, time_steps, num_classes], mean(1) averages over time dimension (dim=1) to get final classification result
        loss = criterion(out, target)

        # measure accuracy and record loss
        prec1 = common.accuracy(out, target, topk=(1,))[0]  # Calculate Top-1 accuracy
        n = images.size(0)  # batch_size, e.g., 256
        losses.update(loss.item(), n)   # losses.avg = weighted average loss of all batches in current epoch
        top1.update(prec1.item(), n)    # top1.avg = weighted average accuracy of all batches in current epoch

        # compute gradient and do SGD step
        optimizer.zero_grad()   # Clear gradients
        loss.backward()         # Backpropagation to calculate gradients
        optimizer.step()        # Update model parameters

        # measure elapsed time
        batch_time.update(time.time() - end)    # Update batch processing time
        end = time.time()                       # Reset time record

        if i % print_freq == 0:
            logger.info(       # Epoch[10](50/200): Loss 0.3245 Prec@1(1) 89.34
                'Epoch[{0}]({1}/{2}): Loss {loss.avg:.4f} Prec@1(1) {top1.avg:.2f}' # Prec@1 = Precision at 1 = Top-1 accuracy
                .format(epoch, i, num_iter, loss=losses,top1=top1))

    scheduler.step()

    return losses.avg, top1.avg # Return average loss and average accuracy for entire epoch

def validate(epoch, val_loader, model, criterion, args):
    batch_time = common.AverageMeter('Time', ':6.3f')
    losses = common.AverageMeter('Loss', ':.4e')
    top1 = common.AverageMeter('Acc@1', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():   # No gradients needed during validation, only forward propagation
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            logits = model(images)
            out = logits.mean(1)
            loss = criterion(out, target)

            # measure accuracy and record loss
            pred1 = common.accuracy(out, target, topk=(1, ))[0]
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))    # Acc@1 91.250

    return losses.avg, top1.avg

def main():
    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)  # Log all command line arguments to log file

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
    logger.info('=== Bit width===:'+str(args.bit))  # === Bit width===:8
    
    # ===== Main modification 2: Create pure_resnet model, then apply quantization pass =====
    # 1. First create pure model (consistent with test_quantize_module_resnet.py)
    model = eval(args.arch)(compress_rate=[0.]*12, num_classes=CLASSES)  # Note: Changed to 12 zeros, consistent with test file
    
    # 2. Set requires_grad (consistent with test_quantize_module_resnet.py)
    for param in model.parameters():
        param.requires_grad = True  # QAT training
    
    # 3. Apply quantization pass (directly copy configuration from test_quantize_module_resnet.py)
    quan_pass_args = {
        "by": "regex_name",
        # Quantize Conv2d layers inside tdLayer (conv1_s, conv2_s), exclude the first conv1
        r"layer\d+\.\d+\.conv[12]_s\.layer\.module$": {
            "config": {
                "name": "rescaw",
                "num_bits": args.bit,  # Use bit from command line arguments
            }
        },
    }
    model, _ = quantize_module_transform_pass(model, quan_pass_args)
    logger.info('==> Applied quantization pass with %d bits' % args.bit)
    
    # 4. Move to device
    model.to(device)
    logger.info(model)  # Output complete model architecture to log

    # calculate model size
    # input_image_size=32
    # input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
    # flops, params = profile(model, inputs=(input_image,))
    # flops, params = clever_format([flops, params], "%.3f")
    # logger.info('Params: %s' % (params))
    # logger.info('Flops: %s' % (flops))

    if len(args.gpu) > 1:
        device_id = []  # device_id = [0,1,2,3] (four GPUs)
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
    optimizer = torch.optim.Adam(
        [{'params': other_parameters},
         {'params': weight_parameters, 'weight_decay': 1e-5}], lr=args.lr, )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    start_epoch = 0
    best_top1_acc= 0

    # load the checkpoint if it exists
    if args.resume:
        checkpoint_dir = os.path.join(args.job_dir, 'checkpoint.pth.tar')
        logger.info('loading checkpoint {} ..........'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']
        # deal with the single-multi GPU problem
        # new_state_dict = OrderedDict()
        # tmp_ckpt = checkpoint['state_dict']
        # if len(args.gpu) > 1:   #情况1: 当前使用多GPU
        #     for k, v in tmp_ckpt.items():   
        #         new_state_dict['module.' + k.replace('module.', '')] = v    #确保所有参数名都有module.前缀
        # else:
        #     for k, v in tmp_ckpt.items():
        #         new_state_dict[k.replace('module.', '')] = v

        # model.load_state_dict(new_state_dict)
        # 直接加载状态字典（GPU环境保持一致）
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("loaded checkpoint {} epoch = {}".format(checkpoint_dir, checkpoint['epoch']))

        # adjust the learning rate according to the checkpoint
        for epoch in range(start_epoch):    # "Catch up" to correct learning rate by calling scheduler.step() in loop
            scheduler.step()
    else:
        logger.info('training from scratch')

    # train the model
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
