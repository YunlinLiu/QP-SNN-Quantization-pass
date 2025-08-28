#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from chop.passes.module.transforms import quantize_module_transform_pass

# Add project root to import pure_resnet model
project_root = Path(__file__).resolve().parents[7]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))
from pure_resnet import resnet_20
from models.layers import LIFSpikeQ


def main():
    # Create model
    resnet = resnet_20(compress_rate=[0.0] * 12, num_classes=10)
    for param in resnet.parameters():
        param.requires_grad = True  # QAT training

    # Q-SNN quantization configuration (weights: first/last 8-bit APoT, hidden 1-bit QSNN; voltage: k-bit via LIFSpikeQ)
    qsnn_pass_args = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q": LIFSpikeQ},
        # first conv (stem)
        r"conv1_s\.layer\.module$": {
            "config": {"name": "apot", "num_bits": 8, "base_k": 2}
        },
        # hidden convs in basic blocks
        r"layer\d+\.\d+\.conv[12]_s\.layer\.module$": {
            "config": {"name": "qsnn"}
        },
        # classifier (last fc)
        r"fc\.module$": {
            "config": {"name": "apot", "num_bits": 8, "base_k": 2}
        },
        # voltage quantization for LIF neurons (stem + blocks)
        r"relu$": {"config": {"name": "lifspike_q", "num_bits": 8, "thresh": 1.0, "tau": 0.5}},
        r"layer\d+\.\d+\.relu[12]$": {"config": {"name": "lifspike_q", "num_bits": 8, "thresh": 1.0, "tau": 0.5}},
    }

    mg, _ = quantize_module_transform_pass(resnet, qsnn_pass_args)
    print(mg)


if __name__ == "__main__":
    main()


