# Posterior Schrödinger Bridges for Continual Learning

Research code and manuscript for **Posterior Schrödinger Bridges (PSB)**,
by Anand Ravishankar and Petar M. Djurić, Stony Brook University.

The method represents accumulated knowledge with a fixed population
of neural networks and a shared diagonal precision. New tasks update this
population through tempered likelihood adaptation and a quadratic constraint
toward each network's carried parameters. The paper develops a Schrödinger
bridge formulation for transport between successive population distributions.

The [manuscript](paper/main.tex) describes the full framework. The runnable
[TPU workflow](psb_tpu/README.md) implements its tempered Adam-plus-proximal
particle-update component. 

## Posterior representation

A particle is the complete parameter vector of one classifier. After task
\(t\), the population contains \(K\) vectors \(\theta^{(t,k)}\).
The paper represents the carried posterior by the Gaussian mixture

$$
\hat q^{(t)}(\theta)
=\frac1K\sum_{k=1}^{K}
\mathcal N\!\left(\theta\mid\theta^{(t,k)},(A^{(t)})^{-1}\right).
$$

The network vectors define component centers. The diagonal precision controls
how strongly subsequent updates constrain displacement in each parameter
direction. The particle count stays fixed throughout the task stream.

Predictions average the classifiers' probabilities,

$$
p(y\mid x)\approx\frac1K\sum_{k=1}^{K}
p(y\mid x,\theta^{(t,k)}).
$$

Training uses the current task's examples. The carried population and precision
summarize earlier tasks for subsequent updates.

## Tempered particle updates

For task \(t+1\), particle \(k\) retains its previous parameter vector as the
anchor \(c=\theta^{(t,k)}\). At temperature \(s\), the intended regularized
objective is

$$
J_s^{(t,k)}(\theta)
=s\,\bar{\mathcal L}^{(t+1)}(\theta)
+\frac12\|\theta-c\|_{R^{(t)}}^2,
$$

where the loss is the mean negative log-likelihood and \(R^{(t)}\) is the
diagonal penalty precision. The default linear schedule uses \(s_m=m/M\),
with \(n\) updates per temperature increment.

Each update computes an Adam direction \(d_r\) from the new-task likelihood
gradients and then applies the quadratic penalty through a proximal step:

$$
\tilde\theta=\theta_r-s_m\eta_r d_r,
\qquad
\theta_{r+1}
=\frac{\tilde\theta+\eta_r R^{(t)}c}
       {1+\eta_r R^{(t)}}.
$$

Products and divisions in the proximal expression are coordinatewise, using
the diagonal entries of the penalty precision. Scaling the Adam displacement
preserves the temperature's effect despite Adam's gradient normalization.
The proximal step contracts displacement from the anchor most strongly in
high-precision directions.

For a fixed Adam direction, the combined update exactly minimizes the local
surrogate

$$
s_m d_r^\top(z-\theta_r)
+\frac{1}{2\eta_r}\|z-\theta_r\|^2
+\frac12\|z-c\|_{R^{(t)}}^2.
$$

This characterizes the adaptive split update. Exactness refers to this local
subproblem, with the history-dependent Adam direction held fixed.

The terminal particle is the network parameter vector after the final update
on the task. Terminal particles become the anchors for the next task and have
uniform weights in the predictive ensemble.

### Precision in the TPU implementation

The TPU implementation normalizes each task's diagonal empirical Fisher by
its coordinate mean and accumulates

$$
h^{(t)}=\gamma h^{(t-1)}
+\frac{F^{(t)}}{\operatorname{mean}(F^{(t)})},
\qquad
R^{(t)}=\lambda\,\operatorname{diag}(h^{(t)}+\epsilon).
$$

The code applies a numerical floor to the normalization denominator.
The configuration fields `lam`, `gamma`, and `floor` specify the penalty
strength, decay, and isotropic floor. This is the implemented regularization
scale. The manuscript's Bayesian precision based on task-size-weighted Fisher
information is a separate modeling definition.

## PSB model

The paper motivates transport between successive posterior approximations
through a Fisher-metric Brownian reference,

$$
d\theta_s=\sigma(A^{(t)})^{-1/2}dW_s.
$$

A continuous endpoint construction uses the carried Gaussian mixture as
\(\mu_t\) and a Gaussian mixture around the updated centers as \(\nu_t\).
The Schrödinger bridge objective is

$$
\min_{\mathbb P}\mathrm{KL}(\mathbb P\|\mathbb Q^{(t)})
\quad\text{subject to}\quad
\mathbb P_0=\mu_t,\qquad\mathbb P_1=\nu_t.
$$

Here the terminal mixture is a constructed approximation to the new posterior.
The tempered objectives guide construction of its centers. Intermediate bridge
marginals are determined by the endpoint laws and reference process.

The fitting stage uses iterative Markovian fitting (IMF), with drift
networks acting on neuron tokens formed from incoming weights and a bias.
The reuse mechanism transfers a fitted drift to guide population
updates at the next task boundary, followed by adaptation to the current task.

## Run the particle-update experiments

The JAX runner batches runs and particles on the available devices. Its default
classifier for Permuted MNIST has two hidden layers of 100 ReLU units and a
shared ten-class output.

Follow the environment instructions in [psb_tpu/README.md](psb_tpu/README.md).
From an environment with the required dependencies installed, inspect the
smoke grid and run it with:

```bash
cd psb_tpu
python -m psb.run grids/smoke.yaml --plan
python -m psb.run grids/smoke.yaml
python -m psb.summarize results/smoke
```

The smoke grid exercises the training paths with a short optimization budget.
The twenty-seed comparison uses baseline hyperparameters selected by the tuning
grid:

```bash
python -m psb.run grids/phase1a_tune.yaml
python -m psb.run grids/phase1b_seeds.yaml
python -m psb.summarize results/phase1b_seeds
```

Use `--results /path/to/results` to choose a results directory. Use the same
directory for tuning and the subsequent comparison so the runner can resolve
the selected baseline hyperparameters.

### Default particle-update configuration

| Setting | Default |
| --- | --- |
| Benchmark | Permuted MNIST, 10 tasks |
| Particles | 128 |
| Hidden layers | 100, 100 |
| Temperature increments | 25 |
| Updates per increment | 100 |
| First-task updates | 3,000 |
| Batch size | 64 |
| Adam learning rate | 0.001 |
| Penalty strength | 4.0 |
| Precision decay | 0.8 |
| Precision floor | 0.001 |
| Fisher example budget | 20,000 |

Grid files override these defaults. Each saved run includes its resolved
configuration.

## Results and repository layout

Each run writes `summary.json` with its configuration, completion status,
accuracy matrix, task history, and final metrics. The final average accuracy is

$$
\mathrm{ACC}^{(T)}=\frac1T\sum_{j=1}^{T}a^{(T,j)}.
$$

The summarizer aggregates completed runs by configuration and reports means,
sample standard deviations, and paired comparisons when a reference is given.

| Path | Contents |
| --- | --- |
| [paper/main.tex](paper/main.tex) | PSB manuscript |
| [psb_tpu/psb/core.py](psb_tpu/psb/core.py) | Batched particle updates, Fisher accumulation, and evaluation |
| [psb_tpu/psb/config.py](psb_tpu/psb/config.py) | Defaults and configuration expansion |
| [psb_tpu/grids](psb_tpu/grids) | Experiment specifications |
| [psb_tpu/psb/summarize.py](psb_tpu/psb/summarize.py) | Aggregation of saved run metrics |
| [bwd_cl](bwd_cl) | PyTorch population and learned-displacement experiments |
| [gwd_cl](gwd_cl) | Earlier weight-diffusion experiments |
| [results_tpu/psb_results](results_tpu/psb_results) | Saved TPU run records |
