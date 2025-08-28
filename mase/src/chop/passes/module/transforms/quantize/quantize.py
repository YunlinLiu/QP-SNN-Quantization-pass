import torch

from chop.nn.quantized.modules import quantized_module_map
from ...module_modify_helper import (
    replace_by_name,
    instantiate_module,
    manual_instantiate_module,
)
from ...state_dict_map import match_a_pattern, check_is_huggingface_model


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def quantize_by_type(network, pass_args):
    for type_name, config in pass_args.items():
        n_m = {}
        for n, m in network.named_modules():
            n_m[n] = m

        if type_name == "linear":
            module = torch.nn.Linear
        elif type_name == "conv2d":
            module = torch.nn.Conv2d
        else:
            raise ValueError(f"{type_name} is not supported!")
        config = config["config"]
        postfix = config.pop("name")
        for n, m in n_m.items():
            if isinstance(m, module):
                new_m = instantiate_module(
                    m, postfix, quantized_module_map, {"config": config}
                )
                network = replace_by_name(network, n, new_m)
    return network


def quantize_by_name(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)
    manual_instantiate = pass_args.get("manual_instantiate", False)
    custom_module_map = pass_args.get("custom_module_map", None)

    quantize_names = pass_args.keys()
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m
    for n, m in n_m.items():
        if n in quantize_names:
            quan_config = pass_args[n]

            quan_config = quan_config["config"]
            postfix = quan_config.pop("name")

            additional_module_args = (
                {"config": quan_config, "network_config": network.config}
                if is_huggingface_model
                else {"config": quan_config}
            )

            try:
                new_m = instantiate_module(
                    m, postfix, quantized_module_map, additional_module_args
                )
            except Exception:
                if not manual_instantiate:
                    raise
                if custom_module_map is None:
                    raise ValueError("manual_instantiate is True but custom_module_map is None")
                # strip 'name' from config for manual instantiation of custom modules
                mi_cfg = {k: v for k, v in quan_config.items() if k != "name"}
                new_m = manual_instantiate_module(
                    m, postfix, custom_module_map, {"config": mi_cfg}
                )
            network = replace_by_name(network, n, new_m)
    return network


def quantize_by_regex_name(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)
    manual_instantiate = pass_args.get("manual_instantiate", False)
    custom_module_map = pass_args.get("custom_module_map", None)

    patterns = list(pass_args.keys())
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m

    for n, m in n_m.items():
        matched_pattern = match_a_pattern(n, patterns)
        if not matched_pattern:
            continue

        quan_config = pass_args[matched_pattern]["config"]
        postfix = quan_config["name"]

        additional_module_args = (
            {"config": quan_config, "network_config": network.config}
            if is_huggingface_model
            else {"config": quan_config}
        )

        try:
            new_m = instantiate_module(
                m, postfix, quantized_module_map, additional_module_args
            )
        except Exception:
            if not manual_instantiate:
                raise
            if custom_module_map is None:
                raise ValueError("manual_instantiate is True but custom_module_map is None")
            mi_cfg = {k: v for k, v in quan_config.items() if k != "name"}
            new_m = manual_instantiate_module(
                m, postfix, custom_module_map, {"config": mi_cfg}
            )
        network = replace_by_name(network, n, new_m)

    return network


def quantize_module_transform_pass(network, pass_args):
    """
    Apply quantization transformation to the given nn.Module.

    :param network: The input network to be transformed.
    :type network: torch.nn.Module

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    Examples pass_args:

    .. code-block:: python

        pass_args = {
            "by": "type", # quantize by type, name, or regex_name
            "default": {"config": {"name": None}}, # default config, this would be used for any node that does not have a specific config
            "linear": {
                "config": {
                    "name": "integer",  # quantization scheme name supported are ["integer", "fixed" (equivalent to integer), "lutnet" (dev mode), "logicnets" (dev mode), "binary", "binary_residual", "ternary", "minifloat_ieee", "minifloat_denorm", "log", "block_fp", "block_minifloat", "block_log"]
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

    :return: The transformed torch.nn.Module.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.

    """
    by = pass_args.pop("by")
    match by:
        case "type":
            network = quantize_by_type(network, pass_args)
        case "name":
            network = quantize_by_name(network, pass_args)
        case "regex_name":
            network = quantize_by_regex_name(network, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')
    return network, {}
