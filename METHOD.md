# Likelihood-Guided Weight Diffusion: mathematical specification

## 1. Continual-learning objective

Let `D_t` be the dataset supplied at task `t`, and let `theta` denote all weights
and biases of one shared classifier. Sequential Bayes gives

```text
p_t(theta) = p(theta | D_1,...,D_t)
           proportional to p(D_t | theta) p_{t-1}(theta).
```

GWD represents `p_{t-1}` with a fixed population of `K` nearby parameter vectors
and a score network. It does not retain any `D_j` after leaving task `j`.

## 2. Establishing the first solution distribution

One MLP is optimized on `D_1` to obtain `theta_star`. For `k=1,...,K`, initialize

```text
theta_1^(k,0) = theta_star + delta_k,
delta_k ~ N(0, rho^2 S^2),
```

where `S` is a block-diagonal tensor scale and antithetic perturbations are used
in pairs. Each copy is then independently refined on `D_1`. Because all copies
originate in the same basin, neuron permutations remain aligned and Euclidean
interpolation between their weights is locally meaningful.

## 3. Parameter normalization and embedding

At the end of task `t`, let the retained parameters be
`Theta_t={theta_t^(1),...,theta_t^(K)}`. Define their mean `mu_t`. Every weight or
bias tensor receives one RMS scale, collected in the diagonal vector `a_t`. The
normalized parameters are

```text
u_t^(k) = (theta_t^(k) - mu_t) / a_t.
```

Let `B_t` contain the leading right singular vectors of the centered matrix whose
rows are `u_t^(k)`. Its rank is at most `K-1`, and

```text
z_t^(k) = B_t u_t^(k)
```

is the low-dimensional parameter embedding. PCA is not used to assert that all
future optima lie in this subspace. It is used only to learn the non-Gaussian
part of the old distribution efficiently.

The old distribution is not defined as `K` Dirac point masses. GWD places an
isotropic Gaussian kernel with fixed standard deviation `tau` around every
retained solution:

```text
q_t(u) = (1/K) sum_k N(u; u_t^(k), tau^2 I).
```

This is essential: `q_t` has full support, so a new likelihood can move mass in
parameter directions absent from the finite sample span. `tau` is one
stream-wide scalar, not a per-task object.

## 4. Forward diffusion and score learning

GWD uses Gaussian, variance-exploding corruption in the embedding. Including
the solution-kernel width, the effective noise is

```text
z_sigma = z_0 + sqrt(tau^2+sigma^2) epsilon,   epsilon ~ N(0,I),
sigma in [sigma_min, sigma_max].
```

A noise-conditional neural network `D_phi(z_sigma,sigma)` predicts `z_0` and is
trained with denoising loss

```text
E_{z_0,sigma,epsilon} ||D_phi(z_0+sigma epsilon,sigma)-z_0||^2_W.
```

Its score estimate follows from Tweedie's identity:

```text
s_phi^z(z_sigma,sigma)
    = (D_phi(z_sigma,sigma)-z_sigma) / sigma^2.
```

There are only `K` clean solutions, but a fresh noise level and Gaussian vector
are sampled on every score-training step. This supplies unlimited corrupted
training pairs; it does not manufacture additional clean modes.

## 5. A full-parameter score

Restricting every transition to the `K-1` dimensional PCA span would prevent a
new task from discovering genuinely new parameter directions. GWD therefore
runs reverse diffusion in the complete normalized parameter space.

Write `u_parallel=B_t^T B_t u` and `u_perp=u-u_parallel`. The clean empirical
distribution has zero orthogonal residual. After isotropic Gaussian smoothing,
its orthogonal score is exactly `-u_perp/(tau^2+sigma^2)`. Hence

```text
s_old(u,sigma)
  = B_t^T s_phi^z(B_t u,sigma) - u_perp/(tau^2+sigma^2).
```

The first term is learned; the second is analytic. This construction preserves
the full parameter space for the new-task likelihood gradient while avoiding a
parameter-sized score network.

## 6. Likelihood-guided reverse diffusion

Before learning task `t+1`, each retained solution undergoes explicit forward
noising:

```text
u_sigma_max^(k) = u_t^(k) + sqrt(tau^2+sigma_max^2) epsilon_k.
```

For classification, the current negative log likelihood is the cross-entropy
`L_{t+1}(theta)`. The score of the sequential posterior is

```text
grad_u log p_{t+1}(u)
  = grad_u log p_t(u) - grad_u L_{t+1}(theta(u)).
```

The original score-addition variant uses the standard guided-diffusion
approximation

```text
s_guided(u,sigma)
  = s_old(u,sigma) - c(sigma) G(grad_u L_{t+1}(theta(u))),
```

where `G` is per-network RMS gradient normalization and `c(sigma)` increases as
noise falls. Guidance is weak when the parameters are strongly corrupted and
becomes strongest when the likelihood gradient is reliable.

A single likelihood term in the reverse score can reconstruct old solutions
without adequately optimizing the incoming task. The default method therefore
uses forward-backward operator splitting. First, it advances the old-posterior
probability flow:

```text
u_prior = OldPosteriorReverseStep(u_i, r_i, r_{i+1}).
```

It then applies an Adam-preconditioned data-consistency step:

```text
u_{i+1} = u_prior - eta_i A_i grad_u L_{t+1}(theta(u_prior)),
```

where `A_i` is the diagonal Adam preconditioner, `eta_i` increases as diffusion
noise falls, and the RMS displacement is bounded. This is a preconditioned
forward-backward approximation to a proximal posterior update. The denoising
operator supplies the old posterior and the data operator supplies the new
likelihood. Unlike ordinary fine-tuning, the old-posterior operator is reapplied
between every pair of likelihood updates.

Because the previous posterior summarizes an increasing number of datasets, its
concentration should increase with the stream length. The implementation uses
the task-level schedule

```text
eta_t = eta_base / (t-1)^a,    t >= 2,
```

with `a=1` by default. This prevents a fixed current-task operator from
overwriting an increasingly informative old posterior. The exponent is shared
across the stream and introduces no task-specific state.

For long streams, this decay is lower-bounded by one stream-wide learning-rate
floor. The floor prevents the likelihood operator from vanishing before a late
task has been acquired; it does not depend on task identity.

### Online curvature geometry

An isotropic data step treats parameters that are essential to old predictions
like parameters lying in flat directions. At the end of each task, GWD estimates
a diagonal empirical Fisher `F_t` from that task while its data are available
and updates one online precision vector

```text
Lambda_t = gamma Lambda_{t-1} + normalize(F_t).
```

Only `Lambda_t` is retained. There is no list of task-indexed Fishers. In the
normalized parameter coordinates `theta=mu+a*u`, the corresponding precision is
proportional to `a^2 Lambda_t`. The proximal likelihood update uses mobility

```text
M_t = normalize((a^2 Lambda_t + delta I)^(-1/2)),
u_{i+1} = u_prior - eta_i M_t A_i grad_u L_{t+1}.
```

RMS normalization keeps the global step budget fixed: curvature reallocates
movement toward directions with low accumulated sensitivity rather than merely
reducing the learning rate. This metric is a stream-wide sufficient statistic
and has `O(P)` memory independent of the number of tasks.

For a descending grid `sigma_i`, define
`r_i=sqrt(tau^2+sigma_i^2)`. The deterministic VE probability-flow update is

```text
u0_hat = u_i + r_i^2 s_old(u_i,r_i),
u_{i+1} = u0_hat + (r_{i+1}/r_i)(u_i-u0_hat).
```

This is a denoising update, not ordinary fine-tuning with noise added. The
old-distribution score participates in every step. The likelihood term bends the
reverse trajectory away from merely reconstructing an old solution and toward
a parameter vector that also explains the current data.

After the last step, the `K` updated networks approximate the new sequential
posterior. They replace the previous population, and the PCA representation and
score network are overwritten. The state is therefore `O(KP)` for `P` classifier
parameters and is independent of the number of tasks.

## 7. Prediction

For input `x`, prediction uses Bayesian-style probability averaging:

```text
p(y | x,D_1:t) approximately (1/K) sum_k p(y | x,theta_t^(k)).
```

The classifier always has one shared ten-way output. There are no task-specific
heads or modules. Permutations are part of the benchmark's input construction,
not learned model components.

## 8. What is exact and what is approximate

Three claims should not be conflated:

1. The forward corruption is an exact Gaussian diffusion.
2. Given an exact old score and exact current likelihood score, their sum is the
   score of the unnormalized sequential posterior at zero noise.
3. At positive noise, using the current likelihood gradient at the noisy
   parameter is an approximation, as in posterior-guided diffusion methods.

The `--score analytic` ablation removes score-network estimation error by using
the exact score of a Gaussian mixture centered on the retained latent solutions.
It does not remove the positive-noise likelihood approximation.
