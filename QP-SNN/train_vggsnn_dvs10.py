#!/usr/bin/env python
"""
训练VGGSNN在DVS-CIFAR10数据集上的脚本
用于复现QP-SNN论文结果
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.quant_vgg import vggsnn
from utils import data_loaders
import utils.common as utils

def main():
    parser = argparse.ArgumentParser(description='Train VGGSNN on DVS-CIFAR10')
    
    # 基本参数
    parser.add_argument('--bit', type=int, default=2, choices=[2, 4, 8],
                        help='量化位宽 (2, 4, 或 8)')
    parser.add_argument('--compress_rate', type=str, default='[0.5,0.5,0.5,0.7,0.7,0.7,0.7,0.0]',
                        help='剪枝率配置')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'prune', 'eval'],
                        help='运行模式: train(训练), prune(剪枝), eval(评估)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减')
    
    # 路径参数
    parser.add_argument('--data_path', type=str, default='/data/dataset/CIFAR10DVS',
                        help='DVS-CIFAR10数据集路径')
    parser.add_argument('--save_dir', type=str, default='./experiments/dvs10_vggsnn',
                        help='保存路径')
    parser.add_argument('--pretrain_path', type=str, default='',
                        help='预训练模型路径')
    
    # 其他参数
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--gpu', type=str, default='0',
                        help='使用的GPU编号')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    save_path = os.path.join(args.save_dir, f'{args.bit}bit_{args.mode}')
    os.makedirs(save_path, exist_ok=True)
    
    # 加载数据集
    print(f'加载DVS-CIFAR10数据集...')
    trainset, testset = data_loaders.build_dvscifar10(path=args.data_path, T=10)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # 解析剪枝率
    compress_rate = eval(args.compress_rate)
    
    # 创建模型
    print(f'创建VGGSNN模型 (量化位宽: {args.bit}bit)...')
    model = vggsnn(compress_rate=compress_rate, num_bits=args.bit, num_classes=10)
    model = model.to(device)
    
    # 加载预训练模型（如果有）
    if args.pretrain_path and os.path.exists(args.pretrain_path):
        print(f'加载预训练模型: {args.pretrain_path}')
        checkpoint = torch.load(args.pretrain_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'预训练模型准确率: {checkpoint.get("best_prec1", "N/A")}%')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.mode == 'train':
        # 训练模式
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                     momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        print('开始训练...')
        best_acc = 0
        for epoch in range(args.epochs):
            # 训练一个epoch
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = torch.mean(outputs, dim=1)  # 时间维度平均
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {loss.item():.4f} '
                          f'Acc: {100.*train_correct/train_total:.2f}%')
            
            # 测试
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    outputs = torch.mean(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            
            test_acc = 100. * test_correct / test_total
            print(f'Epoch {epoch}: Test Accuracy: {test_acc:.2f}%')
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, os.path.join(save_path, 'model_best.pth.tar'))
                print(f'保存最佳模型，准确率: {best_acc:.2f}%')
            
            scheduler.step()
        
        print(f'训练完成！最佳准确率: {best_acc:.2f}%')
    
    elif args.mode == 'prune':
        # 剪枝模式（需要先运行SVS.py计算重要性分数）
        print('剪枝模式：请先运行SVS.py计算重要性分数')
        print('然后使用prun.py进行剪枝和微调')
        
    elif args.mode == 'eval':
        # 评估模式
        model.eval()
        test_correct = 0
        test_total = 0
        
        print('评估模型性能...')
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = torch.mean(outputs, dim=1)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'测试准确率: {test_acc:.2f}%')
        
        # 计算模型大小
        total_params = sum(p.numel() for p in model.parameters())
        model_size = total_params * args.bit / 8 / 1024 / 1024  # MB
        print(f'模型参数量: {total_params:,}')
        print(f'模型大小: {model_size:.2f} MB')

if __name__ == '__main__':
    main()
