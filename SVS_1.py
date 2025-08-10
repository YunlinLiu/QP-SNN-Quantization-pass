import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import data_loaders
import utils.common as utils
from models.quant_vgg import vgg_16_bn, vggsnn
from models.quant_resnet_cifar import resnet_20

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Rank extraction')

parser.add_argument(
    '--dataset',
    default='DVSCIFAR10',
    type=str,
    help='dataset name',
    choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])

parser.add_argument(
    '--arch',
    type=str,
    default='vggsnn',
    choices=('resnet_20','vgg_16_bn'),
    help='The architecture to prune')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='./log/quant/dvs10/2bit/model_best.pth.tar',
    help='load the model from the specified checkpoint')

parser.add_argument(
    '--bit',
    type=int,
    default='2',
    help='Select gpu to use')

parser.add_argument(
    '--limit',
    type=int,
    default=5,
    help='The num of batch to get rank.')

parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch size for training.')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
print('==> Preparing data..')
if args.dataset == 'CIFAR10':
    trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=True, download=False)
    CLASSES = 10
elif args.dataset == 'CIFAR100':
    trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=False, download=False)
    CLASSES = 100
elif args.dataset == 'ImageNet':
    trainset, testset = data_loaders.build_imagenet()
    CLASSES = 1000
elif args.dataset == 'DVSCIFAR10':
    trainset, testset = data_loaders.build_dvscifar10()
    CLASSES = 10
elif args.dataset == 'TinyImageNet':
    trainset, testset = data_loaders.build_tiny_imagenet()
    CLASSES = 200
elif args.dataset == 'DVS128':
    trainset, testset = data_loaders.build_dvs128(T=args.time)
    CLASSES = 11
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)


# Model
print('==> Building model..')
net = eval(args.arch)(compress_rate=[0.0]*100, num_bits=args.bit, num_classes=CLASSES)
net = net.cuda()
print(net)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

if args.pretrain_dir:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.pretrain_dir, map_location='cuda:'+args.gpu)

    from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    #     new_state_dict[k.replace('module.', '',1)] = v
    # net.load_state_dict(new_state_dict)           # 多卡
    net.load_state_dict(checkpoint['state_dict'])   # 单卡
else:
    print('please speicify a pretrain model ')
    raise NotImplementedError

criterion = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total = torch.tensor(0.)


def get_feature_hook(self, input, output):  # use the SVS-based pruning criterion to evaluate kernels
    global feature_result
    global entropy
    global total
    global batch_num
    output = output.mean(1)
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])
    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total

def inference():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    limit = args.limit

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= limit:
               break

            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = criterion(outputs.mean(1), targets)

            test_loss += loss.item()
            _, predicted = outputs.mean(1).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''

if args.arch=='vgg_16_bn' or args.arch=='vggsnn':

    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
    else:
        relucfg = net.relucfg

    for i, cov_id in enumerate(relucfg):
        cov_layer = net.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        if not os.path.isdir('./'+args.arch+'_limit%d'%(args.limit)):
            os.mkdir('./'+args.arch+'_limit%d'%(args.limit))
        np.save('./'+args.arch+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy',
                feature_result.numpy())

        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)