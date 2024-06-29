base_model_path1=./logs_ppo/train_summary_summary/batch_122
base_model_path2=./logs_ppo/train_summary_faithful/batch_122
# base_model_path3=./logs_ppo/train_assistant_humor/batch_100
reward_names="summary,faithful"
exp_type=summary

accelerate launch --main_process_port 29505 eval_rewarded_soups.py --base_model_path1 $base_model_path1 --base_model_path2 $base_model_path2 --reward_names $reward_names --exp_type ${exp_type} --wandb_name $reward_names --save_directory ./logs_rewardedsoups_${exp_type}_eval