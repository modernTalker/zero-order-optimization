#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED="false"
export WANDB_API_KEY=""
export HF_TOKEN="" 

base_command="python run.py"

# Model and Task Configuration
base_command+=" --model_name=\"roberta-large\""
base_command+=" --lora" # type of Fine-Tuning 
base_command+=" --task_name=\"SST2\""
base_command+=" --trainer=\"zo_signsgd\""

# Logging and Reporting
base_command+=" --output_dir=\"result/SST2-FT-\$TAG\""
base_command+=" --report_to=\"wandb\""
base_command+=" --project_name=\"zo-bench\""
base_command+=" --logging_steps=10"

# Training Configuration
base_command+=" --num_train_epochs=5"
base_command+=" --per_device_train_batch_size=16"
base_command+=" --load_best_model_at_end"
base_command+=" --evaluation_strategy=\"steps\""
base_command+=" --save_strategy=\"steps\""
base_command+=" --save_total_limit=1"
base_command+=" --eval_steps=500"
base_command+=" --max_steps=20000"
base_command+=" --save_steps=1000"

# Dataset Settings
base_command+=" --num_eval=1000"
base_command+=" --num_train=1000"
base_command+=" --num_dev=500"
base_command+=" --train_as_classification"
base_command+=" --train_set_seed=0"

# Training Hyperparameters
base_command+=" --perturbation_mode=\"two_side\""
base_command+=" --zo_eps=1e-3"
base_command+=" --momentum=0.0"
base_command+=" --weight_decay=0.0"
base_command+=" --module_wise_perturbation=False"

# Miscellaneous
base_command+=" --overwrite_output_dir"

# Learning Rate Scheduler Settings
base_command+=" --scheduler=\"cosine\""
base_command+=" --num_training_steps=20000"
base_command+=" --warmup_steps=0"
base_command+=" --min_lr_ratio=0.1"
base_command+=" --scheduler_cycle_length=1"

# Sampling Methods
base_command+=" --tensor_sampling_type=\"standard_normal\""
base_command+=" --matrix_sampling_type=\"Random_baseline\""

# Jaguar-Specific Parameters
base_command+=" --zo_tau=1e-3"
base_command+=" --zo_beta=0.9"
# command+=" --zo_use_smoothing=true"

# Sparse Jaguar-Specific Parameters
base_command+=" --params_ratio=0.1"

for learning_rate in 5e-3 1e-2; do
  command="$base_command --learning_rate=${learning_rate}"
  eval "$command"
done
