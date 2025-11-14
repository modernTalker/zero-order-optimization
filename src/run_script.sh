#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="false"
export WANDB_DIR="none"
export WANDB_API_KEY=""
export HF_TOKEN=""

command="python run.py"

# Model and Task Configuration
command+=" --model_name=\"google/gemma-7b\""
# command+=" --lora" # type of Fine-Tuning 
command+=" --task_name=\"SQuAD\""
command+=" --trainer=\"jaguar_signsgd\""
command+=" --max_new_tokens=256"
command+=" --num_return_sequences=1"
command+=" --sampling=True"
# command+=" --temperature=0.2"

# Logging and Reporting
command+=" --output_dir=none"
# command+=" --report_to=\"wandb\""
command+=" --project_name=\"zo-bench\""
command+=" --logging_steps=10"

# Training Configuration
command+=" --num_train_epochs=5"
command+=" --per_device_train_batch_size=4"
command+=" --per_device_eval_batch_size=4"
# command+=" --load_best_model_at_end"
command+=" --eval_strategy=\"steps\""
command+=" --save_strategy=\"no\""
command+=" --save_total_limit=0"
command+=" --eval_steps=500"
command+=" --max_steps=5000"
command+=" --save_steps=0"

# Dataset Settings
command+=" --num_eval=2000" # 33
command+=" --num_train=80000"
# command+=" --num_dev=no"
# command+=" --train_as_classification"
command+=" --train_set_seed=0"

# Training Hyperparameters
command+=" --perturbation_mode=\"two_side\""
command+=" --zo_eps=1e-5"
command+=" --momentum=0.0"
command+=" --weight_decay=0.0"
command+=" --module_wise_perturbation=False"

# Miscellaneous
command+=" --overwrite_output_dir"

# Learning Rate Scheduler Settings
command+=" --learning_rate=1e-6"
command+=" --scheduler=\"constant\""
command+=" --num_training_steps=5000"
command+=" --warmup_steps=0"
command+=" --min_lr_ratio=0.1"
command+=" --scheduler_cycle_length=1"

# Sampling Methods
command+=" --tensor_sampling_type=\"standard_normal\""
command+=" --matrix_sampling_type=\"Torch_QR\""

# Jaguar-Specific Parameters
command+=" --zo_tau=1e-5"
command+=" --zo_beta=0.9"
# command+=" --zo_use_smoothing=true"

# Sparse Jaguar-Specific Parameters
command+=" --params_ratio=0.9"

eval "$command"