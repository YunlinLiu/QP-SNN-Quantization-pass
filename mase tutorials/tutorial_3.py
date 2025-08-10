from transformers import AutoModelForSequenceClassification

from chop import MaseGraph
from chop.tools import get_tokenized_dataset, get_trainer
import chop.passes as passes
from pathlib import Path

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"

mg = MaseGraph(
    model,
    hf_input_names=[
        "input_ids",
        "attention_mask",
        "labels",
    ],
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)

# quantization_config = {
#     "by": "type",
#     "default": {
#         "config": {
#             "name": None,
#         }
#     },
#     "linear": {
#         "config": {
#             "name": "integer",
#             # data
#             "data_in_width": 8,
#             "data_in_frac_width": 4,
#             # weight
#             "weight_width": 8,
#             "weight_frac_width": 4,
#             # bias
#             "bias_width": 8,
#             "bias_frac_width": 4,
#         }
#     },
# }

# mg, _ = passes.quantize_transform_pass(
#     mg,
#     pass_args=quantization_config,
# )

# dataset, tokenizer = get_tokenized_dataset(
#     dataset=dataset_name,
#     checkpoint=tokenizer_checkpoint,
#     return_tokenizer=True,
# )

# trainer = get_trainer(
#     model=mg.model,
#     tokenized_dataset=dataset,
#     tokenizer=tokenizer,
#     evaluate_metric="accuracy",
# )

# trainer.train()
# eval_results = trainer.evaluate()
# print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")

# mg.export(f"{Path(__file__).parent}/tutorial_3_qat")
