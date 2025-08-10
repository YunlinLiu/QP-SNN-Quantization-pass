from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import torch
import chop.passes as passes
from transformers import AutoTokenizer
from chop.tools import get_logger
import torch.fx as fx
from pathlib import Path

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

mg = MaseGraph(model)
mg.draw("bert-base-uncased.svg")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dummy_input = tokenizer(
    [
        "AI may take over the world one day",
        "This is why you should learn ADLS",
    ],
    return_tensors="pt",
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    pass_args={
        "dummy_in": dummy_input,
        "add_value": False,
    },
)

# logger = get_logger("mase_logger")
# logger.setLevel("INFO")


# def count_dropout_analysis_pass(mg, pass_args={}):

#     dropout_modules = 0
#     dropout_functions = 0

#     for node in mg.fx_graph.nodes:
#         if node.op == "call_module" and "dropout" in node.target:
#             logger.info(f"Found dropout module: {node.target}")
#             dropout_modules += 1
#         else:
#             logger.debug(f"Skipping node: {node.target}")

#     return mg, {"dropout_count": dropout_modules + dropout_functions}


# # mg, pass_out = count_dropout_analysis_pass(mg)

# # logger.info(f"Dropout count is: {pass_out['dropout_count']}")

# def remove_dropout_transform_pass(mg, pass_args={}):

#     for node in mg.fx_graph.nodes:
#         if node.op == "call_module" and "dropout" in node.target:
#             logger.info(f"Removing dropout module: {node.target}")

#             # Replace all users of the dropout node with its parent node
#             parent_node = node.args[0]
#             logger.debug(f"This dropout module has parent node: {parent_node}")
#             node.replace_all_uses_with(parent_node)

#             # Erase the dropout node
#             mg.fx_graph.erase_node(node)
#         else:
#             logger.debug(f"Skipping node: {node.target}")

#     return mg, {}


# mg, _ = remove_dropout_transform_pass(mg)
# mg, pass_out = count_dropout_analysis_pass(mg)

# assert pass_out["dropout_count"] == 0

# mg.export(f"{Path(__file__).parent}/tutorial_1")