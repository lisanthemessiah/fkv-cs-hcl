# Kinship Verification with Custom Sampling and Hard Contrastive Loss

This repository contains the code and resources for our paper on facial kinship verification using contrastive learning techniques. Our method combines a custom batch sampling strategy with Hard Contrastive Loss (HCL) to enhance the discriminative power of learned facial features for kinship recognition.

## Abstract

Facial Kinship Verification, a challenging task in computer vision, faces significant challenges due to subtle interclass differences and high intraclass variations. This research introduces a novel approach that leverages supervised contrastive learning techniques with a focus on strategic sample selection. We propose a method that combines a custom batch sampling strategy with Hard Contrastive Loss (HCL) to enhance the discriminative power of learned facial features for kinship recognition.

## Reproducing Results

_WIP. See [Scripts](#Scripts) section for now. Soon I will add instructions for reproducing the results presented in our paper._

To reproduce the results presented in our paper:

1. **Environment Setup**: Instructions for setting up the required environment will be provided upon publication.

2. **Data Preparation**: Details on how to obtain and preprocess the FIW dataset will be included.

3. **Model Training**: 
   - Scripts for training the model with different configurations will be made available.
   - Instructions on how to use the custom batch sampler and implement HCL will be provided.

4. **Evaluation**:
   - Scripts for evaluating the model on kinship verification, tri-subject verification, and search and retrieval tasks will be included.
   - Instructions on how to reproduce the results reported in the paper will be detailed.

5. **Pretrained Models**: 
   - Links to download pretrained models will be provided to allow for quick reproduction of our results.

6. **Ablation Studies**:
   - Scripts and instructions for running the ablation studies reported in the paper will be made available.

Detailed instructions, code, and additional resources necessary for reproducing our results will be added to this repository upon publication of our paper.

## Scripts

This repository includes various utility scripts to help with experiment management, data processing, and result analysis.

### Python Scripts

#### Experiment Analysis
- `fg2025/compute_t1_accuracy.py`: Computes accuracy metrics for Task 1 (kinship verification) from Guild experiment data
- `fg2025/compute_t2t3_accuracy.py`: Computes metrics for Tasks 2 (tri-subject verification) and 3 (search & retrieval)
- `fg2025/kinfacew_results.py`: Processes and analyzes results from KinFaceW dataset experiments

### Shell Scripts

#### Environment Setup
- `setup.sh`: Sets up Python virtual environment using uv and installs dependencies
- `install_zellij.sh`: Installs Zellij terminal multiplexer for improved workflow
- `download_assets.sh`: Downloads required datasets and model weights

#### Experiment Management
- `run_scl_t1.sh`: Runs Task 1 experiments with specified configurations
- `run_scl_t2.sh`: Executes Task 2 (tri-subject verification) experiments
- `run_scl_t3.sh`: Manages Task 3 (search & retrieval) experiments
- `run_kinface.sh`: Handles KinFaceW dataset experiments
- `run_scl_stages.sh`: Orchestrates different stages of SCL model training

#### Guild.ai Utilities
- `guild_queue.sh`: Manages experiment queues across multiple GPUs
- `fix_opref.sh`: Fixes operation references in Guild runs
- `clean_checkpoints.sh`: Cleans up model checkpoints to save disk space
- `restart_experiment.sh`: Automatically restarts stopped or pending experiments

### Usage Examples

#### Setup

```bash
./scripts/shell/setup.sh
```

This script will install the necessary dependencies and set up the virtual environment.

#### Download Assets

```bash
./scripts/shell/download_assets.sh
```

This script will download the necessary repositories, datasets, and model weights.

#### Run experiments

TODO.

## Notebooks

The `notebooks` directory contains three analysis notebooks:

### Analysis Notebooks
- `plot_guild_experiments.ipynb`: Analyzes and visualizes experiment results from Guild.ai runs, including hyperparameter effects and accuracy metrics
- `sampling_analysis.ipynb`: Evaluates the custom kinship sampling strategy, analyzing sampling distributions and scoring patterns
- `tsne.ipynb`: Implements t-SNE visualization for analyzing learned feature embeddings from the model


## Future Work

TODO

## Citation

If you find this work useful in your research, please consider citing:

```
[Citation information will be added upon publication]
```

## Contact

For any questions or concerns, please open an issue in this repository or contact the authors directly.
