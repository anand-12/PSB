# Guided Weight Diffusion for Continual Learning

## Latest iteration: Sequential Posterior-Score Distillation

The recommended method is now `--method spsd`. Instead of discarding the old
score and reconstructing the entire history from only the latest `K` networks,
SPSD trains a single successor score on the sequential Bayesian target
`s_t(theta) = s_{t-1}(theta) - grad_theta L_t(theta)`. An ordinary denoising
term on the newly updated networks supplies current-task support. The old score
is then discarded, so storage remains fixed, replay-free, and task agnostic.

Run the two-seed, two-GPU experiment with:

```bash
cd /export/home/anandr/guided-weight-diffusion-cl
bash scripts/run_spsd_two_gpu.sh
```

This is a new experimental iteration; it has correctness tests but no claimed
accuracy until the full runs complete.

This repository implements **GWD**, a replay-free continual-learning method for
Permuted MNIST. It contains an explicit forward noising process, a learned score
model, and likelihood-guided reverse denoising in neural-network parameter space.

The default classifier is the same-sized MLP commonly used in VCL experiments:
784 inputs, two hidden layers of 100 ReLU units, and one shared 10-class output.
There are 89,610 parameters per network. The method never stores examples and
does not create task-specific heads, adapters, masks, or posterior components.

## What the algorithm does

For Task 1, GWD:

1. trains one classifier;
2. makes `K` copies of its trained parameter vector;
3. independently perturbs the copies; and
4. briefly refines every copy on Task 1.

The resulting `K` nearby networks are treated as samples from a local solution
distribution. Their parameter vectors are normalized tensor-by-tensor and
embedded into a low-dimensional PCA space. A noise-conditional neural network
is trained by denoising score matching on Gaussian-corrupted embeddings.

When Task `t` arrives, the retained networks are explicitly corrupted by the
forward variance-exploding process

```text
u_sigma = u_0 + sqrt(tau^2 + sigma^2) epsilon,   epsilon ~ N(0,I).
```

Here `tau` is a fixed, stream-wide solution-kernel width that gives the old
distribution full support. The process is then reversed. Every reverse step
alternates an old-posterior denoising operator and an Adam-preconditioned
current-task data operator:

```text
u_prior = ReverseStep(u, s_old, sigma)
u_next  = u_prior - eta(sigma) AdamDirection(grad_u L_t(u_prior)).
```

The first operator attracts parameters toward networks that solved Tasks
`1:(t-1)`; the second makes sufficiently strong progress on Task `t`. The
original additive score-guidance update remains available with
`--guidance-mode score` as an ablation.
After reversal, the old score model is discarded and a new one is fitted to the
updated `K` networks. Thus the persistent state is shared across the stream; it
does not grow with the number of tasks.

The proximal data direction is additionally shaped by one online diagonal
Fisher statistic. Parameters that were sensitive on earlier tasks receive small
mobility, while flatter directions receive larger mobility. The statistic is
updated from the current task before its data disappear and accumulated in one
fixed-size vector; no per-task Fisher matrices are retained.

See [METHOD.md](METHOD.md) for the mathematical construction and implementation
details.

## Installation

Use the existing environment that already contains CUDA-enabled PyTorch:

```bash
cd /export/home/anandr/guided-weight-diffusion-cl
python -m pip install -e .
```

## Required first run: three-task posterior calibration

Before another ten-task run, compare two plausible normalized posterior step
sizes on an identical seed and task sequence:

```bash
cd /export/home/anandr/guided-weight-diffusion-cl
bash scripts/run_posterior_probe_two_gpu.sh
```

The two printed records must show that Tasks 2 and 3 are actually acquired; low
forgetting alone is not sufficient. Select the learning rate with the higher
three-task average subject to retaining strong Task-1 accuracy.

## Full run: two seeds on two GPUs

```bash
cd /export/home/anandr/guided-weight-diffusion-cl
bash scripts/run_two_gpu.sh
```

The script downloads MNIST once, launches seed 0 on GPU 0 and seed 1 on GPU 1,
waits for both jobs, and aggregates the final metrics. Its logs are written to
`runs/logs/`; machine-readable summaries are under `runs/paper_seed*/`.

To select different visible devices or Python executable:

```bash
GPU0=2 GPU1=3 PYTHON=/path/to/python bash scripts/run_two_gpu.sh
```

For a short plumbing check before the full job:

```bash
bash scripts/run_smoke_two_gpu.sh
```

The smoke run intentionally uses too few optimization and score-training steps
for a paper result. Do not compare its accuracy with VCL.

## A single run

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m gwd_cl.train \
  --tasks 10 --particles 8 --width 100 --depth 2 \
  --task1-epochs 15 --particle-refine-epochs 3 \
  --score-steps 1500 --reverse-steps 1200 \
  --sigma-max 0.50 --sigma-min 0.005 --solution-kernel-std 0.15 \
  --guidance-mode proximal --posterior-learning-rate 0.03 \
  --posterior-task-decay 1.0 --posterior-min-learning-rate 0.0125 \
  --posterior-max-step 0.05 --seed 0 --tag paper
```

## Required diagnostic ablation

The learned denoiser can be replaced by the exact score of the Gaussian-smoothed
empirical distribution of the `K` latent solutions:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m gwd_cl.train \
  --tasks 10 --particles 8 --score analytic --seed 0 --tag analytic_score
```

This is a diagnostic, not the proposed learned-diffusion result. All other
settings remain the same.

## Outputs

Each task prints one JSON object containing:

- average accuracy over all tasks observed so far;
- average forgetting and backward transfer;
- per-task accuracies and predictive negative log-likelihood;
- denoiser validation loss; and
- forward-noise, likelihood-gradient, and prior-score diagnostics.

Aggregate any set of completed runs with:

```bash
python -m gwd_cl.summarize runs/paper_seed0 runs/paper_seed1
```

## Experimental cautions

- Keep `--permutation-seed` fixed when changing the model seed. Otherwise the
  methods see different task sequences.
- Paper comparisons require matching preprocessing, architecture, task count,
  first-task convention, training data, and evaluation metric. A number copied
  from a paper is not directly comparable unless all six match.
- `K` is a fixed computational/state budget shared by all tasks. It is not the
  number of tasks and does not grow with the stream.
- No implementation can guarantee in advance that a new method beats VCL. The
  included diagnostics are intended to reveal whether failure comes from score
  estimation or likelihood guidance without spending a full sweep budget.

## Background

The implementation follows the variance-exploding score-SDE formulation of
[Song et al.](https://arxiv.org/abs/2011.13456) and the central idea of using a
measurement likelihood to guide reverse diffusion as in
[Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687). Here the
unknown object is a classifier parameter vector and the new-task dataset supplies
the likelihood. The continual-learning construction in this repository is new
experimental code, not an official implementation of either paper.
