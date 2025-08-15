import math
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *

# Key differences from quant_resnet_cifar.py:
# 1. Removed all ReScaWConv quantization logic
# 2. Conv2d is created directly in tdLayer constructor, not saved as independent attribute
#    This avoids duplicate reference issues during quantization pass replacement


def adapt_channel(compress_rate, num_layers):
    """
    Adjust the number of channels in each layer of ResNet according to compression rate
    
    Args:
        compress_rate: List of compression rates for each layer
        num_layers: Total number of network layers (currently only supports 20 layers)
    
    Returns:
        overall_channel: Output channel numbers for each layer
        mid_channel: Middle layer channel numbers in residual blocks
    """
    if num_layers==20:
        # ResNet-20 structure: 3 blocks repeated in each stage
        stage_repeat = [3, 3, 3]
        # Original channel configuration: first layer 64, then 3 layers 128, 3 layers 256, 3 layers 512
        stage_out_channel = [64] + [128] * 3 + [256] * 3 + [512] * 3

    # Calculate output channel compression rate for each layer
    stage_oup_cprate = []
    stage_oup_cprate += [compress_rate[0]]
    for i in range(len(stage_repeat)-1):
        stage_oup_cprate += [compress_rate[i+1]] * stage_repeat[i]
    stage_oup_cprate +=[0.] * stage_repeat[-1]  # Last stage is not compressed
    mid_cprate = compress_rate[len(stage_repeat):]  # Middle layer compression rate

    # Calculate actual channel numbers based on compression rate
    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0 :
            # First layer only has output channels
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
        else:
            # Other layers have both output and middle channels
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i] * (1-mid_cprate[i-1]))]

    return overall_channel, mid_channel


def conv3x3(in_planes, out_planes, stride=1):
    """
    Create 3x3 convolution layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """
    Create 1x1 convolution layer (pointwise convolution)
    Usually used for changing channel numbers or downsampling
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LambdaLayer(nn.Module):
    """
    Lambda layer: Used to perform custom operations in shortcut connections
    Mainly used for padding operations when dimensions don't match
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        # SeqToANNContainer is used to handle time series data
        self.lambd = SeqToANNContainer(lambd)

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    """
    Basic residual block - implements the core structure of SEW ResNet
    
    According to the SEW ResNet paper, this block implements spike-element-wise operations,
    solving gradient vanishing/exploding problems by adding spiking neurons before shortcut connections
    """
    expansion = 1  # Channel expansion factor (BasicBlock does not expand channels)

    def __init__(self, midplanes, inplanes, planes, stride=1):
        """
        Args:
            midplanes: Middle layer channel count (compressed channel count)
            inplanes: Input channel count
            planes: Output channel count
            stride: Stride for downsampling
        """
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        
        # First convolution layer: may perform downsampling
        self.bn1 = tdBatchNorm(midplanes)  # Temporal domain batch normalization
        # Note: Create Conv2d directly in tdLayer, don't save as self.conv1
        # Avoid parameter duplication due to unupdated internal references in tdLayer during quantization pass
        self.conv1_s = tdLayer(conv3x3(inplanes, midplanes, stride), self.bn1)  # Combined as temporal layer

        # First spiking neuron layer (LIF: Leaky Integrate-and-Fire)
        self.relu1 = LIFSpike()

        # Second convolution layer: restore to output channel count
        self.bn2 = tdBatchNorm(planes)
        # Same as above: create Conv2d directly, avoid duplicate reference issues during quantization
        self.conv2_s = tdLayer(conv3x3(midplanes, planes), self.bn2)

        # Second spiking neuron layer (key to implementing SEW ResNet)
        self.relu2 = LIFSpike()
        self.stride = stride

        # Shortcut connection handling
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            # When dimensions don't match, use padding for adjustment
            if stride!=1:
                # Downsampling case: spatial dimensions halved, channel dimensions padded
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],  # Spatial downsampling
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), 
                                    "constant", 0))  # Channel padding            
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
        Forward propagation implementing SEW ResNet block
        
        Key innovations of SEW ResNet:
        1. Don't directly use spiking neurons at the end of main path
        2. First perform residual connection (element-wise addition)
        3. Then pass through spiking neuron
        This avoids gradient vanishing problems in traditional Spiking ResNet
        """
        # First part of main path: convolution -> BN -> spiking neuron
        out = self.conv1_s(x)
        out = self.relu1(out)

        # Second part of main path: convolution -> BN
        out = self.conv2_s(out)

        # SEW operation: first perform residual connection (ADD operation)
        out += self.shortcut(x)
        
        # Then pass through spiking neuron (this is the core improvement of SEW ResNet)
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    """
    Main class for Spiking ResNet network
    
    Implements spiking neural network architecture based on SEW ResNet paper,
    combined with channel pruning functionality
    """
    def __init__(self, block, num_layers, compress_rate, num_classes, step):
        """
        Args:
            block: Basic block type (BasicBlock)
            num_layers: Total number of network layers (e.g., 20)
            compress_rate: List of compression rates for each layer, used for channel pruning
            num_classes: Number of classification categories
            step: Time steps T, simulation time steps for spiking neural network
        """
        super(ResNet, self).__init__()
        # ResNet layer count must satisfy 6n+2 form (e.g., 20=6*3+2)
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6  # Number of blocks per stage

        self.T = step  # Time steps for spiking neural network

        self.num_layer = num_layers
        # Calculate actual channel numbers for each layer based on compression rate
        self.overall_channel, self.mid_channel = adapt_channel(compress_rate, num_layers)
        self.layer_num = 0  # Current layer index

        # First convolution layer (stem layer): RGB image input
        self.bn1 = tdBatchNorm(self.overall_channel[self.layer_num])
        # Note: Create Conv2d directly in tdLayer, don't save as self.conv1
        # This way quantization pass only needs to replace conv1_s.layer.module, avoiding duplicate references
        self.conv1_s = tdLayer(nn.Conv2d(3, self.overall_channel[self.layer_num], 
                               kernel_size=3, stride=1, padding=1, bias=False), self.bn1)  # Wrapped as temporal layer
        self.relu = LIFSpike()  # LIF spiking neuron
        self.layers = nn.ModuleList()
        self.layer_num += 1

        # Build three stages of residual layers
        # First stage: no downsampling, maintain resolution
        self.layer1 = self._make_layer(block, blocks_num=n, stride=1)
        # Second stage: stride=2 for downsampling
        self.layer2 = self._make_layer(block, blocks_num=n, stride=2)
        # Third stage: stride=2 for continued downsampling
        self.layer3 = self._make_layer(block, blocks_num=n, stride=2)

        # Global average pooling and fully connected classifier
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = SeqToANNContainer(nn.Linear(512 * BasicBlock.expansion, num_classes))

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization, suitable for ReLU activation function
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # BN layer initialization: weights to 1, bias to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, blocks_num, stride):
        """
        Build residual layers for one stage
        
        Args:
            block: Residual block type
            blocks_num: Number of blocks in this stage
            stride: Stride of the first block (for downsampling)
        
        Returns:
            Sequential layer containing multiple residual blocks
        """
        layers = []
        # First block may perform downsampling (stride may be 2)
        layers.append(block(self.mid_channel[self.layer_num - 1],  # Middle channel count
                           self.overall_channel[self.layer_num - 1],  # Input channel count
                           self.overall_channel[self.layer_num],  # Output channel count
                           stride))
        self.layer_num += 1

        # Subsequent blocks maintain same resolution (stride=1)
        for i in range(1, blocks_num):
            layers.append(block(self.mid_channel[self.layer_num - 1], 
                               self.overall_channel[self.layer_num - 1],
                               self.overall_channel[self.layer_num]))
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward propagation
        
        Characteristics of spiking neural networks:
        1. Need to repeat input T times in time dimension
        2. Each neuron produces spike output at each time step
        3. Final output is cumulative result of all time steps
        
        Args:
            x: Input image tensor [batch_size, 3, H, W]
        
        Returns:
            Classification output [batch_size, T, num_classes]
        """
        # Add time dimension, replicate input T times: [B, C, H, W] -> [B, T, C, H, W]
        x = add_dimention(x, self.T)
        
        # Stem layer: convolution -> BN -> spiking neuron
        x = self.conv1_s(x)
        x = self.relu(x)

        # Pass through three stages of residual layers
        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        # Global average pooling
        x = self.avgpool(x)
        # Flatten features: [B, T, C, 1, 1] -> [B, T, C]
        x = torch.flatten(x, 2)
        # Fully connected classifier
        x = self.fc(x)
        return x

def resnet_20(compress_rate, num_classes):
    """
    Create ResNet-20 network instance
    
    This is a small ResNet specifically designed for CIFAR datasets,
    combining SEW ResNet architecture with channel pruning
    
    Args:
        compress_rate: List of compression rates for each layer
        num_classes: Number of classification categories (10 for CIFAR-10, 100 for CIFAR-100)
    
    Returns:
        ResNet-20 model instance
    """
    T = 2  # Set time steps to 2 (smaller T can reduce computational load)
    return ResNet(BasicBlock, 20, compress_rate=compress_rate, num_classes=num_classes, step=T)
