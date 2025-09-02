#!/usr/bin/env python3
"""




准备DVS-CIFAR10数据集
"""
import os

import numpy as np
# 兼容 numpy>=2.0 去除 np.bool 的变化
if not hasattr(np, 'bool'):
    np.bool = np.bool_
from spikingjelly.datasets import cifar10_dvs

# 设置数据路径
data_path = '/workspace/QP-SNN-Quantization-pass/data/CIFAR10DVS'

print("开始处理DVS-CIFAR10数据集...")
print(f"数据路径: {data_path}")
print("这可能需要几分钟时间，请耐心等待...")

# 创建数据集实例，这将自动处理已下载的数据
try:
    # 不使用download参数，spikingjelly会自动查找并处理download目录中的数据
    dataset = cifar10_dvs.CIFAR10DVS(
        root=data_path,
        data_type='frame',
        frames_number=10,
        split_by='number'
    )
    print(f"数据集大小: {len(dataset)} 个样本")
    print("数据集准备完成！")
    
    # 打印第一个样本的信息
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"样本形状: {sample[0].shape}")
        print(f"标签: {sample[1]}")
    
except Exception as e:
    print(f"处理数据时出错: {e}")
    print("\n请确保数据文件已经下载到:")
    print(f"  {data_path}/download/")
    print("如果还未下载，请访问: https://www.garrickorchard.com/datasets/cifar10-dvs")
