# Communication Efficient LLM Pre-training with SparseLoCo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![ArXiv](https://img.shields.io/badge/ArXiv-2508.15706-red.svg)](https://arxiv.org/abs/2508.15706)

This repository provides a PyTorch implementation of **SparseLoCo**,  Communication Efficient LLM Pre-training with SparseLoCo. SparseLoCo mitigates the communication bottleneck by combining Top-k EF and DiLoCo.


## Key Features

- **SparseLoCo Optimizer**: A reference implementation of the core algorithm in `src/tplr/sparseloco.py`.
- **Multiple Training Strategies**: Includes baselines for robust comparison:
    - `SparseLoCo`: Proposed Top-k EF compression with local optimization.
    - `DiLoCo`: Distributed training with a local optimization.
    - `DeMo`: Gradient compression with DCT without local optimization.
    - `AdamW`: Standard distributed data-parallel training.

## Getting Started

### 1. Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for environment management)
- This codebase has been tested with H100 and H200 GPUs

### 2. Installation

Clone the repository and install the required dependencies using `uv`.

```bash
git clone https://github.com/tplr-ai/SparseLoCo
cd SparseLoCo
uv sync
source .venv/bin/activate
```

### 3. Data Preparation

The training script expects a pre-tokenized and sharded dataset. Use the `pretokenize_data.py` script to process a dataset from Hugging Face.

The default configuration uses `mlfoundations/dclm-baseline-1.0-parquet` and expects the output in `~/datasets/dclm_tokenized`.

```bash
export DATA_DIR="~/datasets/"
python pretokenize_data.py --output_dir $DATA_DIR/dclm_tokenized
```

*Note: Ensure the `--output_dir` matches the `shards_path` in the sweep configuration files (`hparams/**/*.yaml`) or update the YAML files accordingly.*

## Running Experiments

Experiments are managed through `wandb` sweeps. The `run_sweep.sh` script simplifies the process by creating a sweep and launching a `wandb` agent.

First, set your W\&B API key:

```bash
export WANDB_API_KEY="..."
```

Then, run any of the predefined experiments using the corresponding sweep file. Each experiment is configured to run on **8 GPUs** by default (`--nproc_per_node=8`). You can adjust the number of GPUs by modifying the `--nproc_per_node` parameter in the sweep configuration files.

### SparseLoCo (Proposed Method)

```bash
bash ./run_sweep.sh hparams/512M/sweeps/sparseloco.yaml
```

### Baselines

**DiLoCo Baseline**: Baseline DiLoCo with Nesterov outer optimizer

```bash
bash ./run_sweep.sh hparams/512M/sweeps/diloco_baseline.yaml
```

**DeMo Baseline**: Standard DDP with DeMo

```bash
bash ./run_sweep.sh hparams/512M/sweeps/demo_baseline.yaml
```

**AdamW Baseline**: Standard DDP with AdamW

```bash
bash ./run_sweep.sh hparams/512M/sweeps/adam_baseline.yaml
```

## Citation

If you find **SparseLoCo** useful in your work, please consider citing our work. You can read more the [arXiv preprint](https://arxiv.org/abs/2508.15706).

```bibtex
@misc{sarfi2025sparseloco,
  title        = {Communication Efficient LLM Pre-training with SparseLoCo},
  author       = {Sarfi, Amir and Thérien, Benjamin and Lidin, Joel and Belilovsky, Eugene},
  year         = {2025},
  eprint       = {2508.15706},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  howpublished = {\url{https://arxiv.org/pdf/2508.15706}}
}
```