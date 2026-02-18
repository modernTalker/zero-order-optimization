#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_DISABLED="false"
export WANDB_DIR="none"
export WANDB_API_KEY=""
export HF_TOKEN=""

learning_rates=(2e-6)
trainers=("sgd")

for lr in "${learning_rates[@]}"; do
    for trainer in "${trainers[@]}"; do
        command="python run.py"

        # Model and Task Configuration
        command+=" --model_name=\"gemma_7b\""
        # command+=" --model_name=\"gemma_7b\""
        # command+=" --lora" # type of Fine-Tuning 
        command+=" --task_name=\"HellaSwag\""
        command+=" --trainer=\"$trainer\""
        command+=" --optimizer=\"sgd\""
        command+=" --max_new_tokens=1024"
        command+=" --num_return_sequences=1"
        # command+=" --sampling=True"
        # command+=" --temperature=1"

        # Logging and Reporting
        command+=" --output_dir=none"
        # command+=" --report_to=\"wandb\""
        command+=" --project_name=\"zo-bench\""
        command+=" --logging_steps=10"

        # Training Configuration
        # command+=" --num_train_epochs=2"
        command+=" --per_device_train_batch_size=8"
        command+=" --per_device_eval_batch_size=2"
        # command+=" --load_best_model_at_end"
        # command+=" --load_bfloat16=True"
        # command+=" --eval_strategy=\"steps\""
        command+=" --save_strategy=\"no\""
        command+=" --save_total_limit=0"
        command+=" --eval_steps=100"
        command+=" --max_steps=5000"
        command+=" --save_steps=0"

        # Dataset Settings
        command+=" --num_eval=100"
        command+=" --num_train=117000"
        # command+=" --num_dev=no"
        # command+=" --train_as_classification"
        command+=" --train_set_seed=0"

        # Training Hyperparameters
        command+=" --perturbation_mode=\"two_side\""
        command+=" --zo_eps=1e-3"
        command+=" --momentum=0.9"
        command+=" --weight_decay=0.0"
        command+=" --module_wise_perturbation=False"

        # Miscellaneous
        # command+=" --overwrite_output_dir"

        # command+=" --eval_on_start=True"
        # Learning Rate Scheduler Settings
        command+=" --learning_rate=$lr"
        command+=" --scheduler=\"constant\""
        command+=" --num_training_steps=2000"
        command+=" --warmup_steps=0"
        command+=" --min_lr_ratio=0.1"
        command+=" --scheduler_cycle_length=1"

        # Sampling Methods
        command+=" --tensor_sampling_type=\"standard_normal\""
        command+=" --matrix_sampling_type=\"Torch_QR\""

        # Jaguar-Specific Parameters
        command+=" --zo_tau=1e-3"
        command+=" --zo_beta=0.9"
        # command+=" --zo_use_smoothing=true"

        # Sparse Jaguar-Specific Parameters
        command+=" --params_ratio=0.1"

        eval "$command"
    done
done
