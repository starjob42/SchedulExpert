# SchedulExpert Enhanced

## Overview
**SchedulExpert Enhanced** is an advanced extension of the "Self-Labeling Job Shop Scheduling" framework. This project introduces significant architectural improvements, robust hyperparameter search strategies (SMC, CEM), and enhanced testing utilities to push the state-of-the-art in solving the Job Shop Scheduling Problem (JSP) via self-supervised learning.

---

## Original Work
This repository is a fork and enhancement of:
**[SelfLabelingJobShop](https://github.com/AndreaCorsini1/SelfLabelingJobShop)** by Andrea Corsini et al.

> **Original Citation:**
> *Self-Labeling the Job Shop Scheduling Problem*, Andrea Corsini, Angelo Porrello, Simone Calderara, Mauro Dell'Amico. Arxiv 2024.

We build upon their novel self-supervised training strategy but introduce a more powerful Graph Attention Network (GAT) encoder and a flexible sampling infrastructure.

---

## Key Contributions & Enhancements

We have extended the original repository with the following features:

### 1. Architectural Improvements
**Script:** `train_schedulexpert_improved.py`
- **Mixture of Experts (MoE)**: Integrated into the GATEncoder to drastically increase model capacity without proportional computational cost (`-use_moe`, `-n_experts`).
- **Advanced Encodings**: Added Degree Positional Encodings (`-use_degree_pe`) and flexible Jumping Knowledge (JK) strategies to capture graph topology better.
- **Decoder Enhancements**: Implemented Feature-wise Linear Modulation (FiLM) and Global Attention mechanisms (`-use_film`, `-use_dec_global_attn`) in the MHADecoder.
- **Training Infrastructure**: Fully integrated with **WandB** for experiment tracking and added support for advanced LR schedulers (CosineAnnealing, ReduceLROnPlateau).

### 2. Advanced Hyperparameter Search & Sampling
**Script:** `hyperparam_search_smc_cem.py`
- **Sequential Monte Carlo (SMC)**: Implemented SMC for robust, population-based sampling during inference, allowing the model to escape local optima.
- **Cross-Entropy Method (CEM)**: Added CEM-guided sampling to iteratively refine the sampling distribution towards better solutions.
- **Analysis Tools**: Included `find_best_hparams.py` and clustering/plotting utilities to analyze the trade-offs between wall-time and optimality gaps.

### 3. Improved Testing & Visualization
**Script:** `test_schedulexpert.py`
- **Flexible Testing**: New testing script supporting instance filtering (by job/machine count) and latent space visualization.
- **Benchmarking**: Utilities to compare the improved `SchedulExpert` architecture against baselines.

---

## Results and Performance

We have conducted extensive hyperparameter searches to validate the efficacy of our SMC and CEM sampling methods. The following plots summarize our findings on the dataset:

- **Average Gap Analysis**: [View PDF](hyper_smc_cem_full/summary_avg_gap.pdf) - Shows the average optimality gap across different budgets.
- **Maximum Gap Analysis**: [View PDF](hyper_smc_cem_full/summary_max_gap.pdf) - Highlights the worst-case performance improvements.
- **Wall-time vs Performance**: [View PDF](hyper_smc_cem_full/summary_walltime.pdf) - detailed trade-off analysis between computational budget (time) and solution quality.

> *Note: Click the links above to view the detailed PDF plots located in `hyper_smc_cem_full/`.*

---

## Usage

### Training the Enhanced Model
To train the model with the new architectural features (e.g., MoE, FiLM, Degree PE):
```bash
python train_schedulexpert_improved.py \
    -data_path /path/to/dataset \
    -val_path ./benchmarks/TA \
    -use_moe -n_experts 4 \
    -use_degree_pe \
    -use_film \
    -bs 16 -beta 8
```

### Hyperparameter Search (SMC/CEM)
To run a search comparing different sampling strategies:
```bash
python hyperparam_search_smc_cem.py \
    -model_path checkpoints/your_model.pt \
    -test_path benchmarks/TA \
    -method smc \
    -output_dir output/search_results
```

### Testing and Visualization
To test a trained model and generate results:
```bash
python test_schedulexpert.py \
    -folder_path checkpoints/your_run_name \
    -benchmark TA \
    -num_instances 80 \
    -infer_sch_expert
```

---

## Dependencies
- PyTorch 1.13+
- PyTorch Geometric 2.2+
- WandB (for logging)
- Matplotlib, Seaborn (for plotting)
- Tqdm, Pandas
