#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="false"
# export WANDB_PROJECT="zo-lib-run"       
# export WANDB_ENTITY="andrey"   
export WANDB_API_KEY=""
export HF_TOKEN="" # for llama

command="python run.py"

# Model and Task Configuration
# command+=" --model_name=\"google/gemma-7b\""
command+=" --model_name=\"google/gemma-7b\""
command+=" --load_float16=True"
# command+=" --temperature=0.8"
command+=" --lora" # type of Fine-Tuning 
# command+=" --lora_r=8"
# command+=" --lora_alpha=16"
command+=" --task_name=\"Copa\""
command+=" --trainer=\"jaguar_signsgd\""
# command+=" --optimizer=\"sgd\""

# Logging and Reporting
# TODO: output_dir is constructed in Python using args.tag, do we need it? 
command+=" --output_dir=\"None\""
command+=" --report_to=\"wandb\""
command+=" --project_name=\"zo-bench\""
command+=" --logging_steps=10"

# Training Configuration
command+=" --num_train_epochs=50"
command+=" --per_device_train_batch_size=16"
# command+=" --load_best_model_at_end"
command+=" --eval_on_start=\"True\""
command+=" --eval_strategy=\"steps\""
command+=" --save_strategy=\"no\""
command+=" --save_total_limit=0"
command+=" --eval_steps=1000"
command+=" --max_steps=20000"
command+=" --save_steps=0"

# Dataset Settings
command+=" --num_eval=500"
command+=" --num_train=2000"
command+=" --num_dev=50"
command+=" --train_as_classification"
command+=" --train_set_seed=0"

# Training Hyperparameters
command+=" --perturbation_mode=\"two_side\""
command+=" --zo_eps=1e-4"
command+=" --momentum=0.0"
command+=" --weight_decay=0.0"
command+=" --module_wise_perturbation=False"

# Jaguar-Specific Parameters
command+=" --zo_tau=1e-3"
command+=" --zo_beta=0.9"
command+=" --zo_use_smoothing=true"

# Miscellaneous
command+=" --overwrite_output_dir"

# Learning Rate Scheduler Settings
# command+=" --lr_scheduler_type=\"cosine\"" # FIXME: need to delete this
command+=" --scheduler=\"linear\""
command+=" --num_training_steps=20000"
command+=" --warmup_steps=0"
command+=" --min_lr_ratio=0.1"
command+=" --scheduler_cycle_length=1"

# Sampling Methods
command+=" --vector_sampling_type=\"standard_normal\""
command+=" --matrix_sampling_type=\"Torch_QR\""

# Learning Rate Loop
for learning_rate in 1e-4; do
    full_command="$command --learning_rate=$learning_rate"
    eval "$full_command"
done
