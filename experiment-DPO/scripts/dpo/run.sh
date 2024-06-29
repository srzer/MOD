LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=3 --main_process_port 29501"

divergence_type="reverse_kl"
sft_model_name="PKU-Alignment/alpaca-7b-reproduced"
prompt_template="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K"
preference="better"
sanity_check=False
output_dir="./output"
max_length=256
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=2
learning_rate=5e-4
dpo_run_name="${dataset_name}/dpo-${divergence_type}/dpo-${preference}"
dpo_model_name="${output_dir}/${dpo_run_name}/merged_checkpoint"
adapter_model_name="${output_dir}/${dpo_run_name}/adapter_checkpoint.bin"

PYTHONPATH=. $LAUNCH scripts/dpo/dpo.py \
    --divergence_type ${divergence_type} \
    --sft_model_name ${sft_model_name} \
    --prompt_template "${prompt_template}" \
    --dataset_name "${dataset_name}-${preference}" \
    --sanity_check ${sanity_check} \
    --max_length ${max_length} \
    --training_args.output_dir "${output_dir}/${dpo_run_name}" \
    --training_args.run_name ${dpo_run_name} \
    --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
    --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
    --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
    --training_args.learning_rate ${learning_rate} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0