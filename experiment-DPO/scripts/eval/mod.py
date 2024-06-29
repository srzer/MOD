import os
from dataclasses import dataclass, field
from typing import Optional
import random
import torch
from torch.utils.data import Dataset, DataLoader
import tyro
from accelerate import Accelerator
from peft import LoraConfig
import sys
sys.path.append(".")
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification
import os
from tqdm import tqdm
from src.trainer.light_dpo_trainer import DPOTrainer_Light
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import print_local_main, disable_progress_bar_non_local_main, prepare_model_for_peft, param_sharding_enabled, set_seed
from src.utils.util_decode import FusionModel

os.environ["WANDB_MODE"] = "dryrun"
disable_progress_bar_non_local_main()

@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "the sft model name"})
    dpo_model_1_name: str = field(metadata={"help": "the dpo model 1 name"})
    dpo_model_2_name: str = field(metadata={"help": "the dpo model 2 name"})
    dpo_model_3_name: str = field(default=None, metadata={"help": "the dpo model 3 name"})
    weight_1: float = field(default=0.5, metadata={"help": "the weight for dpo model 1"})
    weight_2: float = field(default=0.5, metadata={"help": "the weight for dpo model 2"})
    weight_3: float = field(default=0, metadata={"help": "the weight for dpo model 3"})
    num_beams: int = field(default=1, metadata={"help": "the number of beams"})
    seed: int = field(default=42, metadata={"help": "the seed"})
    f_type: str = field(default="reverse_kl")
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "whether to conduct sanity check"})

    beta: Optional[float] = field(default=0.1, metadata={"help": "beta for kl control"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    num_proc: Optional[int] = field(default=4, metadata={"help": "num_proc for dataset.map"})
    generate_during_eval: Optional[bool] = field(default=True, metadata={"help": "whether to generate during evaluation"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/dpo",
            overwrite_output_dir=True,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,
            weight_decay=0.05,
            fp16=True,
            remove_unused_columns=False,
            run_name="dev_dpo",

            num_train_epochs=3,
            logging_steps=10,
            save_steps=0.25,
            eval_steps=0.25,
            eval_delay=0.25,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
        )
    )

    peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft for training"})
    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    )

script_args = tyro.cli(ScriptArguments)
if not script_args.peft:
    script_args.peft_config = None

set_seed(script_args.seed)

# base model
print_local_main("loading model...")
sft_model = AutoModelForCausalLM.from_pretrained(
    script_args.sft_model_name,
    use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
    torch_dtype=torch.bfloat16, # necessary for llama2, otherwise will be cast to float32
    **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
)
sft_model.config.update({
    "use_cache": True,
    "pad_token_id": sft_model.config.eos_token_id 
})
sft_model = prepare_model_for_peft(sft_model, peft_config=script_args.peft_config, args=script_args.training_args)
sft_model.load_adapter(script_args.dpo_model_1_name, "model_0")
sft_model.load_adapter(script_args.dpo_model_2_name, "model_1")
if script_args.dpo_model_3_name is not None and script_args.weight_3 > 0:
    sft_model.load_adapter(script_args.dpo_model_3_name, "model_2")
    sample_model = FusionModel(sft_model, [script_args.weight_1, script_args.weight_2, script_args.weight_3], f_type=script_args.f_type)
else:
    sample_model = FusionModel(sft_model, [script_args.weight_1, script_args.weight_2], f_type=script_args.f_type)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# dataset
if not script_args.dataset_caching:
    from datasets import disable_caching
    disable_caching()
rdp = DATASET_CONFIGS[script_args.dataset_name](
    prompt_template=script_args.prompt_template,
    sanity_check=script_args.sanity_check,
)
train_dataset = rdp.get_preference_dataset(split="train")
eval_dataset  = rdp.get_preference_dataset(split="validation")

trainer = DPOTrainer_Light(
    ref_model=sft_model,
    beta=script_args.beta,
    args=script_args.training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=script_args.max_length,
    num_proc=script_args.num_proc,
    generate_during_eval=script_args.generate_during_eval,
)
eval_dataloader = trainer.get_eval_dataloader(trainer.eval_dataset)

num_samples = len(trainer.eval_dataset)
iters = 50
results = []

for _ in tqdm(range(iters)):
    random_indices = random.sample(range(num_samples), k=trainer.args.eval_batch_size)
    print("sampled: "+str(len(random_indices))+"/"+str(num_samples))
    dataloader = trainer.get_eval_dataloader(trainer.eval_dataset)
    random_batch_dataset = dataloader.dataset.select(random_indices)
    # random_eval_dataloader = trainer.get_eval_dataloader(random_batch_dataset)
    batch = trainer.data_collator(random_batch_dataset, generate=True)
    batch = trainer._prepare_inputs(batch)

    prompt_input_ids = batch["prompt_input_ids"]
    prompt_attention_mask = batch["prompt_attention_mask"]
    
    policy_output = sample_model.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        max_length=script_args.max_length,
        do_sample=False,
        num_beams=script_args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    for idx in range(prompt_input_ids.shape[0]):
        prompt = tokenizer.decode(prompt_input_ids[idx], skip_special_tokens=True)
        policy_output_decoded = tokenizer.decode(policy_output[idx], skip_special_tokens=True)
        results.append(policy_output_decoded)

output_prefix = "" if script_args.num_beams == 1 else str(script_args.num_beams)+"_"

dataset_prefix = ""
if "PKU" in script_args.dataset_name:
    dataset_prefix = "_beavertail"
elif "UltraFeedback" in script_args.dataset_name:
    dataset_prefix = "_ultrafeedback"
elif "HelpSteer" in script_args.dataset_name:
    dataset_prefix = "_helpsteer"
elif "summarize" in script_args.dataset_name:
    dataset_prefix = "_summarize"
else: raise NotImplementedError

file_path = "results{}/outputs/{}output_{}_{}_{}.txt".format(dataset_prefix, output_prefix, script_args.weight_1, script_args.weight_2, script_args.f_type)

if not os.path.exists(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w") as f:
    for result in results:
        f.write("\nPrompt and response\n")
        f.write(result)
        f.write("\n")