# Leveraging Coordinate Momentum in SignSGD and Muon: Memory-Optimized Zero-Order LLM Fine-Tuning

This repository contains the code for experiments applying Jaguar SignSGD, Jaguar Muon and ZO-Muon methods for different LLM Fine-Tuning tasks.

All the code is located in the `src` folder.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

To train and evaluate the model in the paper, run this command:

```
./run_script.sh
```

## Methods 

* `jaguar_muon` is Jaguar Muon
* `jaguar_signsgd` is Jaguar SignSGD
* `zo_muon` is ZO-Muon
