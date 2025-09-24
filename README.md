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

* `zo_ns_jaguar` is Jaguar Muon
* `zo_jaguar` is Jaguar SignSGD
* `zo_muon` is ZO-Muon
