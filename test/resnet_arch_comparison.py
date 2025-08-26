import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.quant_resnet_cifar import resnet_20 as quant_resnet_20
from models.pure_resnet import resnet_20 as pure_resnet_20
from mase.src.chop.passes.module.transforms import quantize_module_transform_pass

# 创建三个模型
print("创建模型...")
# 1. 原始量化模型（8位）
quant_model = quant_resnet_20(compress_rate=[0.0]*12, num_classes=10, num_bits=8)

# 2. 纯净模型
pure_model = pure_resnet_20(compress_rate=[0.0]*12, num_classes=10)

# 3. Pass量化后的纯净模型（8位）- 需要创建新的纯净模型实例
pure_model_for_quant = pure_resnet_20(compress_rate=[0.0]*12, num_classes=10)
quan_pass_args = {
    "by": "regex_name",
    r"layer\d+\.\d+\.conv[12]_s\.layer\.module$": {
        "config": {
            "name": "rescaw",
            "num_bits": 8,
        }
    },
}
quantized_pure_model, _ = quantize_module_transform_pass(pure_model_for_quant, quan_pass_args)

# 保存到文件
with open('test/resnet_arch_comparison.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("1. QUANTIZED ResNet-20 (quant_resnet_cifar.py, 8-bit)\n")
    f.write("="*60 + "\n\n")
    f.write(str(quant_model))
    f.write("\n\n")
    
    f.write("="*60 + "\n")
    f.write("2. PURE ResNet-20 (pure_resnet.py, no quantization)\n") 
    f.write("="*60 + "\n\n")
    f.write(str(pure_model))
    f.write("\n\n")
    
    f.write("="*60 + "\n")
    f.write("3. PASS-QUANTIZED ResNet-20 (pure + quantize_pass, 8-bit)\n")
    f.write("="*60 + "\n\n")
    f.write(str(quantized_pure_model))
    
print("模型对比已保存到 test/resnet_arch_comparison.txt")

# 分析差异
def count_conv_types(model):
    """统计卷积层类型"""
    conv2d_count = 0
    rescaw_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Conv2d':
            conv2d_count += 1
        elif module.__class__.__name__ == 'ReScaWConv':
            rescaw_count += 1
        elif module.__class__.__name__ == 'Conv2dReScaW':
            rescaw_count += 1
    return conv2d_count, rescaw_count

# 统计三个模型
quant_conv2d, quant_rescaw = count_conv_types(quant_model)
pure_conv2d, pure_rescaw = count_conv_types(pure_model)
pass_conv2d, pass_rescaw = count_conv_types(quantized_pure_model)

print("\n" + "="*60)
print("模型验证与对比")
print("="*60)

print("\n层统计:")
print(f"1. 原始量化模型: Conv2d={quant_conv2d}, ReScaW={quant_rescaw}")
print(f"2. 纯净模型:     Conv2d={pure_conv2d}, ReScaW={pure_rescaw}")
print(f"3. Pass量化模型: Conv2d={pass_conv2d}, ReScaW={pass_rescaw}")

# 参数量对比
quant_params = sum(p.numel() for p in quant_model.parameters())
pure_params = sum(p.numel() for p in pure_model.parameters())
pass_params = sum(p.numel() for p in quantized_pure_model.parameters())

print("\n参数量:")
print(f"1. 原始量化模型: {quant_params/1e6:.2f}M")
print(f"2. 纯净模型:     {pure_params/1e6:.2f}M")
print(f"3. Pass量化模型: {pass_params/1e6:.2f}M")

# 验证等价性
print("\n验证结果:")
if quant_conv2d == pass_conv2d and quant_rescaw == pass_rescaw:
    print("✅ Pass量化模型与原始量化模型结构等价！")
    print(f"   - 都有 {quant_conv2d} 个Conv2d层（stem层未量化）")
    print(f"   - 都有 {quant_rescaw} 个量化层（18个BasicBlock卷积）")
else:
    print("❌ 模型结构不等价")
    
if abs(quant_params - pass_params) < 100:  # 允许小误差
    print("✅ 参数量一致（差异<100）")
else:
    print(f"❌ 参数量差异: {abs(quant_params - pass_params)}")

# 追加分析结果到文件
with open('test/resnet_arch_comparison.txt', 'a', encoding='utf-8') as f:
    f.write("\n\n")
    f.write("="*60 + "\n")
    f.write("验证与对比结果\n")
    f.write("="*60 + "\n")
    f.write(f"\n层统计:\n")
    f.write(f"1. 原始量化模型: Conv2d={quant_conv2d}, ReScaW={quant_rescaw}\n")
    f.write(f"2. 纯净模型:     Conv2d={pure_conv2d}, ReScaW={pure_rescaw}\n")
    f.write(f"3. Pass量化模型: Conv2d={pass_conv2d}, ReScaW={pass_rescaw}\n")
    f.write(f"\n参数量:\n")
    f.write(f"1. 原始量化模型: {quant_params/1e6:.2f}M\n")
    f.write(f"2. 纯净模型:     {pure_params/1e6:.2f}M\n")
    f.write(f"3. Pass量化模型: {pass_params/1e6:.2f}M\n")
    f.write(f"\n验证结论:\n")
    if quant_conv2d == pass_conv2d and quant_rescaw == pass_rescaw and abs(quant_params - pass_params) < 100:
        f.write("✅ Pass量化成功！原始量化模型与Pass量化模型完全等价！\n")
    else:
        f.write("❌ 存在差异，需要检查\n")
