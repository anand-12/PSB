# Annealed Predictive Transport for Continual Learning

Working proposal, 2026-09-17. Target: ICLR 2027 (abstract Sep 18 AoE, paper Sep 25 AoE).

## 1. The one change that makes the bridge non-vacuous

Version 0 bridged between two endpoints that were both fixed before the bridge
ran, so only the coupling and the path were free, and both turned out empty
(coupling = one permutation, target ≈ source). The fix is to let `beta` index
the **target**, not the path:

    pi_beta(F) ∝ q_{t-1}(F) · L_t(F)^beta,      beta in [0, 1]

where `F` are centred logits at `M` inducing inputs, `q_{t-1}` is the previous
functional posterior at those points, and `L_t` is the current-task likelihood.
`beta = 0` is the old functional posterior; `beta = 1` is the new one. Each
stage now has a different target, so schedules, resampling and transport all
have something to do.

Three endpoint bugs from the v0 study are fixed by this move:

1. **Likelihood scale.** v0 used `--target-likelihood-scale 40` over 64 probes,
   i.e. 2,560 pseudo-observations against 60,000 real ones, which is why the
   target only reached 31-47% accuracy on the probes it was conditioned on. The
   principled value is `N_t / S` (about 937 for 64 probes), recovering the
   actual posterior.
2. **Hand-set loss weights.** `old_weight = 5.0`, `current_weight = 1.0` are
   replaced by the tempered path, which sets the trade-off by `beta`.
3. **Identity coupling.** v0 initialised each target chain at its own source
   particle, so optimal transport just undid the shuffle. SMC resampling
   between stages destroys particle identity, so the coupling carries
   information.

## 2. Method

State: `K` particles of function values `F^k in R^{M x C}` at the inducing set
(fixed size, task-balanced, labels never retained), realised by a population of
networks as in the v0 code.

For each boundary `t`, over a schedule `0 = beta_0 < ... < beta_S = 1`:

```
Algorithm 1: Annealed Predictive Transport (one task boundary)
Input: particles F^{1:K} from task t-1, inducing set Z, current task data D_t
 1: fit q_{t-1} to F^{1:K}            # Gaussian (diagonal + low rank) at inducing points
 2: for s = 1..S do
 3:   w^k ∝ L_t(F^k)^{beta_s - beta_{s-1}}                    # SMC incremental weights
 4:   if ESS(w) < tau: resample particles                      # breaks particle identity
 5:   T_s <- transport from {F^k, w^k} to pi_{beta_s}          # the bridge (below)
 6:   F^k <- T_s(F^k)
 7:   distil: train network population on KL(F^k || f_theta(Z)) + CE on D_t batches
 8: end for
```

Step 5 is where the Schrödinger bridge sits, and where the ablation axis is:

- **(b) Entropic bridge (proposed).** Sinkhorn coupling between the weighted
  cloud at `beta_{s-1}` and a sample of `pi_{beta_s}`, then a Brownian bridge in
  the inducing-value space. This is the v0 machinery, now between genuinely
  different distributions. Keep the marginal-preserving sampling, but replace
  Hungarian + Birkhoff with a Gumbel-Sinkhorn or barycentric projection so the
  step stays inside XLA (no host round-trip on TPU).
- **(a) Langevin only.** Standard SMC move step; the zero-transport control.
- **(c) Learned flow.** AFT/CRAFT-style trained transport, if time allows.

**Residual decomposition (requested).** Split `F = mean + residual`. The mean
follows the tempered path deterministically; the bridge transports only the
ensemble spread. v0 already implements this (`residual_bridge_targets`), and it
becomes meaningful once the endpoints differ. Ablation: mean-only, residual-only,
both.

**beta schedules (requested).** linear; geometric; power `beta = (s/S)^p`;
adaptive by ESS/CESS; and a **per-inducing-point schedule** that anneals
current-task probes faster than old-task probes (the principled version of v0's
`--endpoint-mix`).

## 3. Why this could beat the direct control

The direct control distils to a single endpoint with fixed loss weights. The
annealed path (i) sets the stability/plasticity trade-off by the likelihood
itself rather than a tuned constant, (ii) keeps the distillation target close to
the network's current function at every stage, which is an easier optimisation
problem (the lifting error in v0 was 4e-3 to 1e-2 KL and grew with memory
pressure), and (iii) maintains a particle approximation of the functional
posterior instead of collapsing it, which is where the v0 five-task NLL/ECE
advantage came from.

## 4. Positioning

- **Function-space CL**: VCL (Nguyen et al., 2018), FROMP (Pan et al., 2020),
  S-FSVI (Rudner et al., 2022). These are the baselines and the closest prior
  work; all use inducing points, none uses an annealed transport between
  consecutive functional posteriors.
- **Annealed transport**: AFT (Arbel et al., 2021), CRAFT (Matthews et al.,
  2022), SCLD (ICLR 2025), Adjoint Schrödinger Bridge Sampler (NeurIPS 2025).
  Machinery we adapt; none is applied to continual learning.
- **Risk citation**: "On Sequential Bayesian Inference for Continual Learning"
  (Kessler et al.) reports that even near-exact sequential Bayes fails to prevent
  forgetting in BNNs. We must address this head-on: our claim is that the
  approximation is much better conditioned at inducing points in function space
  than in weight space.

Contribution claim: continual learning as a sequence of tempered function-space
posteriors, transported by an entropic bridge and distilled into the network;
plus the schedule/residual analysis and a TPU implementation.

## 5. Experiments

Benchmarks (chosen for 8-day feasibility and for direct comparability with the
Bayesian-CL baselines): Split-MNIST, Permuted-MNIST (10 tasks), Split-CIFAR-10,
Split-CIFAR-100 with precomputed frozen features. Metrics: average accuracy,
forgetting, NLL, ECE, and backward transfer; 3 seeds minimum.

Baselines: v0 direct endpoint distillation (the control that already ties),
fine-tuning, EWC, LwF, FROMP, S-FSVI (published numbers where protocols match).

Ablations: transport (entropic bridge / Langevin only / deterministic OT);
beta schedule (5 variants above); residual decomposition (3 variants);
resampling on/off; particle count K in {4, 8, 16, 32}; likelihood scale.

**Kill gate (must run first).** Permuted-MNIST 10 tasks, 3 seeds: the annealed
path must beat v0 direct distillation by at least 0.5 pp average accuracy, or
clearly win on NLL/ECE at parity. v0 numbers to beat: 0.9062 accuracy, 0.0659
forgetting, 0.4556 NLL, 0.1739 ECE. If the gate fails, the paper pivots to the
analysis/negative-result framing in section 7.

## 6. TPU plan

JAX + Flax + Optax on a TPU VM (v3-8 or v4-8). The particle population is a
single `[K, P]` array as in v0, so `vmap` over particles and `jit` over the
whole boundary update. Constraints: no `scipy.optimize.linear_sum_assignment`
and no Birkhoff loop (host round-trips); use log-domain Sinkhorn, fixed shapes,
and `lax.scan` over the beta schedule. Data through `tensorflow_datasets`.
The v0 PyTorch code does not port; the JAX rewrite is roughly 600 lines.

## 7. Fallback framing if the kill gate fails

"When does transport help in function-space continual learning?" — a rigorous
analysis paper: the tempered path as the principled replacement for tuned
distillation weights, the collapse of the coupling to a permutation under
locally-initialised posteriors, the likelihood-scale bug, and the conditions
(particle count, endpoint separation, memory pressure) under which stochastic
transport can matter at all. Weaker, but publishable and honest, and it uses
every run already on disk.

## 8. Draft title and abstract for registration

**Title**: Annealed Predictive Transport: Continual Learning as Sequential
Function-Space Sampling

**Abstract** (draft):
Continual learning is sequential Bayesian inference, but practical methods
approximate it with a single distillation target and hand-tuned loss weights.
We recast each task boundary as a sequence of tempered posteriors over function
values at a fixed set of inducing inputs, interpolating from the previous
functional posterior to the one implied by the new task's likelihood. A
population of networks is carried along this path by an entropic transport step
with marginal-preserving couplings, and distilled back into weights at every
stage. The tempering parameter replaces the usual stability/plasticity
constants, resampling between stages keeps the particle approximation of the
predictive posterior from collapsing, and a mean/residual decomposition lets the
stochastic component act only on epistemic spread. We study which parts matter:
annealing schedules, the entropic transport step against its deterministic and
Langevin-only limits, and the number of particles. Experiments on
Split-MNIST, Permuted-MNIST and Split-CIFAR compare against function-space
continual learning baselines and against a matched direct-distillation control,
reporting accuracy, forgetting, and calibration.

(Abstract is deliberately claim-neutral so the result can be written up either way.)
