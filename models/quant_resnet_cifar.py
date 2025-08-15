import math
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from models.quant_function import ReScaWConv

def adapt_channel(compress_rate, num_layers):
    """
    根据压缩率调整ResNet每层的通道数
    
    Args:
        compress_rate: 各层的压缩率列表
        num_layers: 网络总层数（目前仅支持20层）
    
    Returns:
        overall_channel: 各层输出通道数
        mid_channel: 残差块中间层通道数
    """
    if num_layers==20:
        # ResNet-20的结构：每个阶段重复3个块
        stage_repeat = [3, 3, 3]
        # 原始通道配置：第一层64，然后3层128，3层256，3层512
        stage_out_channel = [64] + [128] * 3 + [256] * 3 + [512] * 3

    # 计算每层的输出通道压缩率
    stage_oup_cprate = []
    stage_oup_cprate += [compress_rate[0]]
    for i in range(len(stage_repeat)-1):
        stage_oup_cprate += [compress_rate[i+1]] * stage_repeat[i]
    stage_oup_cprate +=[0.] * stage_repeat[-1]  # 最后一个阶段不压缩
    mid_cprate = compress_rate[len(stage_repeat):]  # 中间层压缩率

    # 根据压缩率计算实际通道数
    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0 :
            # 第一层只有输出通道
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
        else:
            # 其他层有输出通道和中间通道
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i] * (1-mid_cprate[i-1]))]

    return overall_channel, mid_channel


def conv3x3(in_planes, out_planes, stride=1, num_bit=None):
    """
    创建3x3量化卷积层
    使用ReScaWConv实现权重量化，支持指定量化位数
    """
    return ReScaWConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, num_bits=num_bit)

def conv1x1(in_planes, out_planes, stride=1, num_bit=None):
    """
    创建1x1量化卷积层（点卷积）
    通常用于改变通道数或下采样
    """
    return ReScaWConv(in_planes, out_planes, kernel_size=1, stride=stride, num_bits=num_bit)


class LambdaLayer(nn.Module):
    """
    Lambda层：用于在shortcut连接中执行自定义操作
    主要用于维度不匹配时的padding操作
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        # SeqToANNContainer用于处理时间序列数据
        self.lambd = SeqToANNContainer(lambd)

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    """
    基础残差块 - 实现SEW ResNet的核心结构
    
    根据SEW ResNet论文，这个块实现了spike-element-wise操作，
    通过在shortcut连接前添加脉冲神经元来解决梯度消失/爆炸问题
    """
    expansion = 1  # 通道扩展倍数（BasicBlock不扩展通道）

    def __init__(self, midplanes, inplanes, planes, stride=1, num_bit=None):
        """
        Args:
            midplanes: 中间层通道数（压缩后的通道数）
            inplanes: 输入通道数
            planes: 输出通道数
            stride: 步长，用于下采样
            num_bit: 量化位数
        """
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        
        # 第一个卷积层：可能进行下采样
        self.conv1 = conv3x3(inplanes, midplanes, stride, num_bit)
        self.bn1 = tdBatchNorm(midplanes)  # 时域批归一化
        self.conv1_s = tdLayer(self.conv1, self.bn1)  # 组合为时域层

        # 第一个脉冲神经元层（LIF: Leaky Integrate-and-Fire）
        self.relu1 = LIFSpike()

        # 第二个卷积层：恢复到输出通道数
        self.conv2 = conv3x3(midplanes, planes,num_bit=num_bit)
        self.bn2 = tdBatchNorm(planes)
        self.conv2_s = tdLayer(self.conv2, self.bn2)

        # 第二个脉冲神经元层（实现SEW ResNet的关键）
        self.relu2 = LIFSpike()
        self.stride = stride

        # Shortcut连接处理
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            # 当维度不匹配时，使用padding调整
            if stride!=1:
                # 下采样情况：空间维度减半，通道维度padding
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],  # Spatial downsampling
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), 
                                    "constant", 0))  # Channel padding            
            # [batch, channel, height:step:2, width:step:2], take every other pixel, 32×32 → 16×16
            # Channel padding: F.pad parameters, result: pad 64 zero channels before and after 128 channels, total 256 channels.
            else:
                # Only channel count mismatch: only perform channel padding
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), 
                                    "constant", 0))
            
            # Alternative: use 1x1 convolution to adjust dimensions (commented out)
            '''self.shortcut = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                #nn.BatchNorm2d(planes),
            )#'''

    def forward(self, x):
        """
        前向传播实现SEW ResNet块
        
        SEW ResNet的关键创新：
        1. 在主路径末尾不直接使用脉冲神经元
        2. 先进行残差连接（element-wise addition）
        3. 然后再通过脉冲神经元
        这避免了传统Spiking ResNet的梯度消失问题
        """
        # 主路径第一部分：卷积 -> BN -> 脉冲神经元
        out = self.conv1_s(x)
        out = self.relu1(out)

        # 主路径第二部分：卷积 -> BN
        out = self.conv2_s(out)

        # SEW操作：先进行残差连接（ADD操作）
        out += self.shortcut(x)
        
        # 然后通过脉冲神经元（这是SEW ResNet的核心改进）
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    """
    量化脉冲ResNet网络主类
    
    实现了基于SEW ResNet论文的量化脉冲神经网络架构，
    结合了权重量化和通道剪枝功能
    """
    def __init__(self, block, num_layers, compress_rate, num_classes, num_bits, step):
        """
        Args:
            block: 基础块类型（BasicBlock）
            num_layers: 网络总层数（如20）
            compress_rate: 各层压缩率列表，用于通道剪枝
            num_classes: 分类类别数
            num_bits: 权重量化位数
            step: 时间步数T，脉冲神经网络的仿真时间步
        """
        super(ResNet, self).__init__()
        # ResNet的层数必须满足6n+2的形式（如20=6*3+2）
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6  # 每个阶段的块数

        self.T = step  # 脉冲神经网络的时间步数
        self.num_bits = num_bits  # 量化位数

        self.num_layer = num_layers
        # 根据压缩率计算各层的实际通道数
        self.overall_channel, self.mid_channel = adapt_channel(compress_rate, num_layers)
        # 不剪枝时：
        # overall_channel = [64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        # mid_channel = [128, 128, 128, 256, 256, 256, 512, 512, 512]
        # ResNet-20 遵循 6n+2 的设计原则（n=3），共20层：
        # 层数计算：
        # - 1个Stem卷积层 = 1层
        # - 9个残差块 × 2个卷积/块 = 18层  
        # - 1个全连接层 = 1层
        # 总计 = 20层
        self.layer_num = 0  # 当前层索引

        # 第一个卷积层（stem层）：RGB图像输入
        self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = tdBatchNorm(self.overall_channel[self.layer_num])
        self.conv1_s = tdLayer(self.conv1, self.bn1)  # 包装为时域层
        self.relu = LIFSpike()  # LIF脉冲神经元
        self.layers = nn.ModuleList()
        self.layer_num += 1

        # 构建三个阶段的残差层
        # 第一阶段：不下采样，保持分辨率
        self.layer1 = self._make_layer(block, blocks_num=n, stride=1)
        # 第二阶段：stride=2进行下采样
        self.layer2 = self._make_layer(block, blocks_num=n, stride=2)
        # 第三阶段：stride=2继续下采样
        self.layer3 = self._make_layer(block, blocks_num=n, stride=2)

        # 全局平均池化和全连接分类器
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = SeqToANNContainer(nn.Linear(512 * BasicBlock.expansion, num_classes))

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,ReScaWConv):
                # Kaiming初始化，适用于ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # BN层初始化：权重为1，偏置为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, blocks_num, stride):
        """
        构建一个阶段的残差层
        
        Args:
            block: 残差块类型
            blocks_num: 该阶段包含的块数
            stride: 第一个块的步长（用于下采样）
        
        Returns:
            包含多个残差块的Sequential层
        """
        layers = []
        # 第一个块可能进行下采样（stride可能为2）
        layers.append(block(self.mid_channel[self.layer_num - 1],  # 中间通道数
                           self.overall_channel[self.layer_num - 1],  # 输入通道数
                           self.overall_channel[self.layer_num],  # 输出通道数
                           stride, self.num_bits))
        self.layer_num += 1

        # 后续块保持相同分辨率（stride=1）
        for i in range(1, blocks_num):
            layers.append(block(self.mid_channel[self.layer_num - 1], 
                               self.overall_channel[self.layer_num - 1],
                               self.overall_channel[self.layer_num], 
                               num_bit=self.num_bits))
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        
        脉冲神经网络的特点：
        1. 需要在时间维度上重复输入T次
        2. 每个神经元在每个时间步产生脉冲输出
        3. 最终输出是所有时间步的累积结果
        
        Args:
            x: 输入图像张量 [batch_size, 3, H, W]
        
        Returns:
            分类输出 [batch_size, T, num_classes]
        """
        # 添加时间维度，将输入复制T次：[B, C, H, W] -> [B, T, C, H, W]
        x = add_dimention(x, self.T)
        
        # Stem层：卷积 -> BN -> 脉冲神经元
        x = self.conv1_s(x)
        x = self.relu(x)

        # 通过三个阶段的残差层
        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        # 全局平均池化
        x = self.avgpool(x)
        # 展平特征：[B, T, C, 1, 1] -> [B, T, C]
        x = torch.flatten(x, 2)
        # 全连接分类器
        x = self.fc(x)
        return x

def resnet_20(compress_rate, num_classes, num_bits):
    """
    创建ResNet-20网络实例
    
    这是一个专门为CIFAR数据集设计的小型ResNet，
    结合了SEW ResNet架构、权重量化和通道剪枝
    
    Args:
        compress_rate: 各层压缩率列表
        num_classes: 分类类别数（CIFAR-10为10，CIFAR-100为100）
        num_bits: 权重量化位数
    
    Returns:
        ResNet-20模型实例，参数量约0.46M
    """
    T = 2  # 设置时间步数为2（较小的T可以减少计算量）
    return ResNet(BasicBlock, 20, compress_rate=compress_rate, num_classes=num_classes, num_bits=num_bits, step=T)
# ResNet forward过程：
# [B, C, H, W]                    # 原始输入
#     ↓ add_dimention
# [B, T, C, H, W]                  # 添加时间维度
#     ↓ conv1_s (tdLayer)
# [B, T, C', H, W]                 # SeqToANNContainer处理
#     ↓ relu (LIFSpike)  
# [B, T, C', H, W]                 # LIFSpike保持维度
#     ↓ BasicBlock
# [B, T, C'', H', W']              # 通过残差块
#     ↓ avgpool
# [B, T, C'', 1, 1]                # 全局池化
#     ↓ flatten
# [B, T, C'']                      # 展平空间维度
#     ↓ fc
# [B, T, num_classes]              # 最终输出

# 输入 [B,3,32,32]
#     ↓
# Conv2d (未量化) → 64通道
#     ↓
# Stage 1: 3个BasicBlock (6个量化Conv)
#     ↓
# Stage 2: 3个BasicBlock (6个量化Conv) + 下采样
#     ↓  
# Stage 3: 3个BasicBlock (6个量化Conv) + 下采样
#     ↓
# 全局池化
#     ↓
# Linear (未量化) → num_classes