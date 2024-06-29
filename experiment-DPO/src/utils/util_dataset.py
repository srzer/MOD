import torch
from src.trainer.dpo_trainer import DPODataMapFunc, DPODataCollatorWithPadding

def relabel_dataset(eval_dataset, tokenizer, rm_1, rm_2, weight_1, weight_2):
    for i in range(len(eval_dataset)):
        sample = eval_dataset[i]
        chosen_prompt = sample['chosen']
        rejected_prompt = sample['rejected']

        chosen_inputs = tokenizer([chosen_prompt], return_tensors='pt', padding=True, truncation=True).to(rm_1.device)
        rejected_inputs = tokenizer([rejected_prompt], return_tensors='pt', padding=True, truncation=True).to(rm_1.device)

        with torch.no_grad():
            chosen_scores = weight_1*rm_1(**chosen_inputs)[0] + weight_2*rm_2(**chosen_inputs)[0]
            rejected_scores = weight_1*rm_1(**rejected_inputs)[0] + weight_2*rm_2(**rejected_inputs)[0]

        if chosen_scores > rejected_scores:
            eval_dataset[i]['chosen'] = chosen_prompt
            eval_dataset[i]['rejected'] = rejected_prompt
        else:
            eval_dataset[i]['chosen'] = rejected_prompt
            eval_dataset[i]['rejected'] = chosen_prompt
    return eval_dataset
  
def preprocess_dataset(dataset, tokenizer, max_length, label_pad_token_id=-100, num_proc=4, filter_too_long=True):
    # tokenize samples
    dataset = dataset.map(
        DPODataMapFunc(tokenizer, label_pad_token_id=label_pad_token_id), 
        batched=True, 
        num_proc=num_proc, 
        remove_columns=dataset.column_names
    )
    original_length = len(dataset)
    # filter samples that are too long
    if filter_too_long:
        dataset = dataset.filter(
            lambda x: len(x["prompt_chosen_input_ids"]) <= max_length and len(x["prompt_rejected_input_ids"]) <= max_length
        )
    else:
        dataset = dataset.map(
            # truncate chosen and rejected
            lambda sample: {k: v[:max_length] if ('chosen' in k or 'rejected' in k) else v for k, v in sample.items()},
            num_proc=num_proc,
        )
    filtered_length = len(dataset)
    data_collator = DPODataCollatorWithPadding(tokenizer, label_pad_token_id=label_pad_token_id)
    return dataset, filtered_length / original_length, data_collator
