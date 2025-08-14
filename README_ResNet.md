# ResNet-20 训练配置说明

## 文件说明

- `quant_resnet.py`: ResNet-20在CIFAR-10上的训练脚本
- `models/quant_resnet_cifar.py`: ResNet模型定义（基于SEW ResNet架构）
- `run_resnet.sh`: 运行训练的Shell脚本

## 主要特性

1. **SEW ResNet架构**: 实现了Spike-Element-Wise ResNet，解决了传统Spiking ResNet的梯度消失问题
2. **权重量化**: 支持不同位数的权重量化（默认8位）
3. **通道剪枝**: 支持通过compress_rate参数进行通道剪枝
4. **脉冲神经网络**: 使用LIF神经元，时间步T=2

## 运行方法

### 基础训练
```bash
python quant_resnet.py
```

### 使用Shell脚本
```bash
chmod +x run_resnet.sh
./run_resnet.sh
```

### 命令行参数
```bash
python quant_resnet.py \
    --arch resnet_20 \      # 模型架构
    --dataset CIFAR10 \     # 数据集
    --batch_size 128 \      # 批大小
    --epochs 300 \          # 训练轮数
    --lr 0.001 \           # 学习率
    --bit 8 \              # 量化位数
    --job_dir ./log_resnet/ \  # 日志目录
    --workers 4            # 数据加载线程数
```

### 从检查点恢复训练
```bash
python quant_resnet.py --resume
```

## 配置差异（相比VGG）

| 参数 | VGG | ResNet |
|-----|-----|---------|
| batch_size | 256 | 128 |
| architecture | vgg_16_bn | resnet_20 |
| log_dir | ./log/ | ./log_resnet/ |
| workers | 8 | 4 |

## 模型特点

- **层数**: 20层（适用于CIFAR-10的小型ResNet）
- **通道数**: 16 -> 32 -> 64（三个阶段）
- **残差连接**: SEW方式（先Add后Spike）
- **时间步**: T=2（脉冲神经网络）

## 输出文件

训练后会在`./log_resnet/`目录下生成：
- `logger*.log`: 训练日志
- `checkpoint.pth.tar`: 最新检查点
- `model_best.pth.tar`: 最佳模型
- `config.txt`: 配置记录

## 注意事项

1. 确保已安装所需依赖
2. 首次运行会自动下载CIFAR-10数据集
3. GPU内存建议至少4GB
4. 训练时间约需几小时（取决于GPU性能）
