from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the adapter name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the merged model name"})
    dtype: Optional[str] = field(default="fp16", metadata={"help": "dtype"})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "the merged model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to get"
assert script_args.output_name is not None, "please provide the output name of the Adapter"

str2dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float}

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
if peft_config.task_type == "SEQ_CLS":
    # The sequence classification task is used for the reward model in PPO
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.adapter_model_name, num_labels=1, torch_dtype=str2dtype[script_args.dtype]
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.adapter_model_name, return_dict=True, torch_dtype=str2dtype[script_args.dtype]
    )

# Load the PEFT model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

adapter_state_dict = model.state_dict()
filtered_model_dict = {name: model_obj for name, model_obj in adapter_state_dict.items() if 'lora' in name.lower()}
print(filtered_model_dict.keys())
torch.save(adapter_state_dict, f"{script_args.output_name}")