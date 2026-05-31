# Zero-order Fine-tuning

This is a repository for running and comparing fine-tuning methods and for adding custom optimizers (with emphasis on zero-order optimizers). This README contains: project overview, installation, usage, examples, how to add a custom optimizer, an example `run_script.sh`, and recommended practices.

## Table of contents

- [Project overview](#project-overview)
- [Features](#features)
- [Supported fine-tuning methods](#supported-fine-tuning-methods)
- [Datasets](#datasets)
- [Models](#models)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [run_script.sh (example)](#run_scriptsh-example)
- [CLI arguments](#cli-arguments)
- [Common examples](#common-examples)
- [How to add a custom optimizer](#how-to-add-a-custom-optimizer)
- [Minimal optimizer template](#minimal-optimizer-template)
- [Optimizer registry example](#optimizer-registry-example)
- [Trainer selection example](#trainer-selection-example)
- [Perturbation utilities](#perturbation-utilities)
- [Available optimizers](#available-optimizers)
- [Logging and checkpoints](#logging-and-checkpoints)
- [Tests and reproducibility](#tests-and-reproducibility)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project overview

This repository contains code to run fine-tuning experiments, evaluate different fine-tuning strategies on standard NLP tasks, and extend the optimizer API with custom optimizers — including a focus on zero-order optimization methods (ZO). The training pipeline handles data loading, model preparation, selection of fine-tuning method, optimization loop, logging and checkpointing. The codebase is designed to make it easy to add new optimizers and compare them against first-order baselines.

## Features

- Multiple fine-tuning methods: `full_ft`, `lora`, `prefix`
- Support for common NLP datasets: `SST2`, `COPA`, `WinoGrande`
- Pluggable optimizer API: add an optimizer by subclassing `ZeroOrderOptimizer`
- Parameter perturbation utilities for zero-order gradient estimation (e.g. `matrix_pertrub_params`)
- CLI-driven experiments and reproducible checkpoints
- Example scripts and registry-based optimizer selection

## Supported fine-tuning methods

- `full_ft`
- `lora`
- `prefix`

## Datasets

- `SST2`
- `COPA`
- `WinoGrande`

## Models

- `facebook/opt-1.3b`
- `roberta-large`
- `facebook/opt-13b`
- `llama-7b`
- `qwen`

## Repository layout

TODO 


## Requirements

- Python 3.10
- Conda (recommended) or virtualenv
- CUDA-compatible GPU for large-model experiments (optional)
- Hugging Face credentials / token for gated models when required

## Installation

From repository root

```bash
cd src
conda create -n ZOLLM python=3.10 -y
conda activate ZOLLM
pip install -r requirements.txt
```

## Quickstart

From src/ run the provided wrapper:

```bash 
bash run_script.sh
```

### run_script.sh (example)

Save the following as src/run_script.sh and make it executable with chmod +x src/run_script.sh

```bash 
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="false"
export WANDB_API_KEY="YOUR_API_KEY"
export HF_TOKEN="" # if needed

command="python run.py"

# Model and Task Configuration
command+=" --model_name=\"roberta-large\""
command+=" --lora" # type of Fine-Tuning 
command+=" --task_name=\"SST2\""
command+=" --trainer=\"jaguar_signsgd\""

# Logging and Reporting
command+=" --output_dir=\"result/SST2-FT-\$TAG\""
command+=" --report_to=\"wandb\""
command+=" --project_name=\"zo-bench\""
command+=" --logging_steps=10"

# Training Configuration
command+=" --num_train_epochs=5"
command+=" --per_device_train_batch_size=16"
command+=" --load_best_model_at_end"
command+=" --evaluation_strategy=\"steps\""
command+=" --save_strategy=\"steps\""
command+=" --save_total_limit=1"
command+=" --eval_steps=1000"
command+=" --max_steps=20000"
command+=" --save_steps=1000"

# Dataset Settings
command+=" --num_eval=1000"
command+=" --num_train=1000"
command+=" --num_dev=500"
command+=" --train_as_classification"
command+=" --train_set_seed=0"

# Training Hyperparameters
command+=" --learning_rate=1e-3"
command+=" --perturbation_mode=\"two_side\""
command+=" --zo_eps=1e-3"
command+=" --momentum=0.0"
command+=" --weight_decay=0.0"
command+=" --module_wise_perturbation=False"

# Miscellaneous
command+=" --overwrite_output_dir"

# Learning Rate Scheduler Settings
command+=" --scheduler=\"constant\""
command+=" --num_training_steps=20000"
command+=" --warmup_steps=0"
command+=" --min_lr_ratio=0.1"
command+=" --scheduler_cycle_length=1"

# Sampling Methods
command+=" --vector_sampling_type=\"standard_normal\""
command+=" --matrix_sampling_type=None"

# Jaguar-Specific Parameters
command+=" --zo_tau=1e-3"
command+=" --zo_beta=0.9"

eval "$command"
```

## CLI arguments

Common arguments that should be supported and documented in `run.py`:

```bash 
TODO
```

## Common examples
Full fine-tuning baseline on RoBERTa-Large
```bash 
TODO
```

## How to add a custom optimizer

Add a new file `src/optimizers/<your_optimizer>.py`.

Implement a class that inherits from `ZeroOrderOptimizer` and implements required methods (e.g., step, plus any repo-specific API such as perturb / sample_vector if present).

Export the class in `src/optimizers/__init__.py`.

Register the optimizer name in the optimizer registry and ensure `run.py` / `trainer.py` can instantiate it via `--trainer`.

Run training with `--trainer <your_name>`.

## Minimal optimizer template

Create `src/optimizers/my_zo.py` with the following content

```python 
TODO
```

## Optimizer registry example

Edit `src/optimizers/__init__.py` as follows

```python 
from .zo_muon import ZO_MUON
from .zo_sampling_muon import ZO_SamplingMUON
from .jaguar_muon import Jaguar_MUON
from .jaguar_signsgd import Jaguar_SignSGD
from .zo_sgd import ZO_SGD
from .zo_signsgd import ZO_SignSGD
from .zo_adam import ZO_Adam
from .zo_conserv import ZO_Conserv

# which optimizers will be added by calling *
__all__ = [
    'ZO_MUON', 'ZO_SamplingMUON', 'Jaguar_MUON', 'Jaguar_SignSGD', 
    'ZO_SGD', 'ZO_SignSGD', 'ZO_Adam', 'ZO_Conserv'
]
```

## Trainer selection example

In `trainer.py` instantiate chosen optimizer from registry

```python 
if args.trainer == "YOUR_OPTIMIZER":
    self.optimizer = YOUR_OPTIMIZER(params)
```

## Perturbation utilities

TODO (type of perturbation and type of sampling?)

Parameter perturbation for zero-order gradient estimation is provided by `TensorSampler` in `src/optimizers/opt_utils/tensor_sampling.py`. Make sampling distribution, sampling budget, and perturbation scale configurable via CLI or config files. Typical options:


* vector sampling type: standard_normal, etc.
* matrix sampling type: None or structured designs

## Available optimizers

* `ZO_SGD`

* `ZO_Adam`

* TODO

First-order baselines via wrappers: `adam`, `sgd`

Update this list with short descriptions as new optimizers are added.

## Logging and checkpoints

Save the best checkpoint by validation metric to `--save_dir`.

Save the full CLI command and config alongside each checkpoint.

Support optional experiment logging backends (CSV, Weights & Biases) using `--report_to` and environment variables.

## Tests and reproducibility

Use `--train_set_seed` to improve reproducibility.

Add unit tests for optimizer behavior and perturbation utilities (pytest).

Include examples/ with reproducible commands and minimal configs.

## Contributing

1. Fork the repository
2. Add optimizer / dataset / method
3. Add tests and example commands
4. Open a PR with a clear description and example run
