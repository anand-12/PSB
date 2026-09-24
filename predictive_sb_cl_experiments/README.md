# Inducing Predictive Schrödinger Bridges

This directory is a self-contained falsification experiment for predictive-space
Schrödinger bridges in continual learning. It does not modify or import the
existing weight-diffusion implementation.

At each task boundary, an ensemble induces a distribution of centered logits on
a fixed-size input memory. A component-wise functional Bayesian update produces
the target predictive distribution. Sinkhorn scaling computes the empirical
Schrödinger coupling in a whitened predictive PCA space. A sampled
Birkhoff--von Neumann permutation preserves both finite endpoint marginals, and
a coherent Brownian bridge trajectory is distilled back into the network
population.

The core comparison is deliberately matched:

- `direct`: optimal endpoint pairing and direct endpoint distillation;
- `random_path`: deterministic path with a random endpoint coupling;
- `ot_path`: deterministic conditional-mean path with optimal assignment;
- `sb`: Sinkhorn coupling and stochastic Brownian bridge path;
- `finetune`: no inducing-point preservation.

All methods use the same architecture, current-task batches, update budget, and
(except `finetune`) predictive posterior endpoint samples.

## Run

```bash
cd predictive_sb_cl_experiments
/export/home/anandr/miniforge3/envs/test2/bin/python -m pytest -q
sh scripts/run_smoke.sh
sh scripts/run_core_ablation.sh
sh scripts/run_bridge_variants.sh
sh scripts/run_replicates_and_long.sh
```

The implementation makes a deliberately narrow claim: Sinkhorn plus conditional
Brownian bridges is an exact empirical SB in the finite predictive embedding;
distillation into neural weights is an approximate lifting whose endpoint error
is reported explicitly.

See [`RESULTS.md`](RESULTS.md) for the completed initial falsification study.
The present evidence supports predictive inducing-point regularisation, but it
does **not** show an accuracy advantage from the stochastic bridge over a
matched direct-distillation control.
