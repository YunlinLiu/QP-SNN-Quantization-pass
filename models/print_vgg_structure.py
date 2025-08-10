#!/usr/bin/env python3
"""
SNN VGG16 网络结构打印脚本
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.quant_vgg import VGG

# 创建模型
model = VGG(compress_rate=[0.0] * 12, num_bits=4, num_classes=10, step=4)

# 打印网络结构
print(model)

# 统计参数
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数数量: {total_params:,}")
