#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from chop.passes.module.transforms import quantize_module_transform_pass

project_root = Path(__file__).resolve().parents[7]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))
from pure_vgg import vgg_16_bn
from models.layers import LIFSpikeQ


def main():
    vgg = vgg_16_bn(compress_rate=[0.0] * 16, num_classes=10)
    vgg.T = 2

    pass_args = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q": LIFSpikeQ},
        r"features\.convbn0\.layer\.module": {
            "config": {"name": "apot", "num_bits": 8, "base_k": 2}
        },
        r"features\.convbn(?!0\b)\d+\.layer\.module": {
            "config": {"name": "qsnn"} 
        },
        r"classifier\.linear1\.module": {
            "config": {"name": "apot", "num_bits": 8, "base_k": 2}
        },
        r"features\.relu\d+": {
            "config": {"name": "lifspike_q", "num_bits": 2, "thresh": 1.0, "tau": 0.5}
        },
    }

    mg, _ = quantize_module_transform_pass(vgg, pass_args)

    print(mg)


if __name__ == "__main__":
    main()


