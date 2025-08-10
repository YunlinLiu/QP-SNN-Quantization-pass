import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
from chop import MaseGraph
import chop.passes as passes

# Create VGG-16 model, adapted for CIFAR-10
model = vgg16_bn(weights=None)
model.classifier[6] = nn.Linear(4096, 10)


# Create MaseGraph
mg = MaseGraph(model)
print(model)

# Generate graph visualization
mg.draw("vgg16_torchvision.svg")


# Create CIFAR-10 format dummy input
dummy_input = torch.randn(1, 3, 32, 32)

# Initialize metadata analysis
mg, _ = passes.init_metadata_analysis_pass(mg)

# Add common metadata analysis
mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    pass_args={
        "dummy_in": {"x": dummy_input},
        "add_value": False,
    },
)

# Quantization configuration (similar to Tutorial 3)
quantization_config = {
    "by": "type",
    "default": {
        "config": {
            "name": None,
        }
    },
    "conv2d": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

# Apply quantization transformation
mg, _ = passes.quantize_transform_pass(
    mg,
    pass_args=quantization_config,
)

mg.draw("vgg16_torchvision_quantized.svg")
