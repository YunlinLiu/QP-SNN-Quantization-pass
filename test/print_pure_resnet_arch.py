import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pure_resnet import resnet_20

model = resnet_20(compress_rate=[0.0]*12, num_classes=10)

# 保存到文件
with open('test/print_pure_resnet_arch.txt', 'w') as f:
    f.write(str(model))
    
print("模型结构已保存到 test/print_pure_resnet_arch.txt")
