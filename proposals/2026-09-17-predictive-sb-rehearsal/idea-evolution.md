# Idea evolution: Schrödinger bridges for continual learning

## Version 0 (2026-09-17) — restructured from the predictive-SB falsification study

### Why the original formulation was restructured

The study in `predictive_sb_cl_experiments/` bridged between two predictive
endpoints that were already fixed (ensemble logits and a Langevin functional
posterior started from those same logits). Forgetting is measured at the
endpoint, so the bridge could only change the coupling and the path:

- the coupling collapsed to one permutation (normalised entropy exactly 0.5,
  one Birkhoff component), so SB = deterministic OT = particle-wise distillation;
- the target barely moved (old-probe KL ~1e-4), so the path had nothing to
  interpolate;
- every path-only variant (SGD-aligned reference, residual bridge, MC paths,
  temperature) tied direct distillation within seed noise.

Design rules carried forward:

1. The bridge must determine an endpoint that distillation cannot reach on its
   own, not interpolate between endpoints that are already known.
2. The coupling must be between unpaired distributions; with given pairs it is
   regression.
3. Every stochastic component is ablated against its zero-noise limit from the
   first experiment.
4. Evaluation uses a current benchmark with strong exemplar-free baselines.

### Research Idea Brief

- **Problem Statement**: In exemplar-free class-incremental learning with an
  adapting backbone, nothing constrains the new network on past-task inputs.
  Current remedies either move feature statistics without samples (SDC, LDC,
  EFC, AdaGauss) or synthesise pseudo-inputs that are adversarial, off-manifold
  or low-diversity (DeepInversion, ABD, the adversarial transport in ADC).
- **Proposed Approach**: Define past-class pseudo-data as the endpoint of a
  Schrödinger half-bridge. The reference process is a Brownian/OU diffusion
  started at current-task inputs x ~ q; the terminal potential is the previous
  model's Bayesian posterior predictive for an old class c. The half-bridge
  endpoint is pi_c(x) ∝ q_sigma(x) · pbar_old(c | x)^beta. Samples come from the
  h-transform-guided SDE (optionally an amortised drift network). The new model
  distils the old posterior predictive on bridged samples and uses them for
  head balancing and drift compensation.
- **Key Innovation**:
  1. The bridge supplies an endpoint that is otherwise missing (where to
     preserve the old predictive), satisfying design rule 1.
  2. Guidance by a posterior predictive (head ensemble or last-layer Laplace)
     rather than a point network, expected to give less adversarial, more
     semantic pseudo-samples.
  3. Entropic regularisation controls pseudo-sample diversity; ADC-style
     adversarial transport is the zero-noise, point-estimate limit, which gives
     a clean ablation axis (to be made precise).
  4. No discrete assignment (Hungarian/Birkhoff), so the method is XLA/TPU-native.
- **Target Domain**: continual learning, exemplar-free class-incremental vision.
- **Target Venue**: ICLR 2027 (abstract Sep 18 2026 AoE, paper Sep 25 2026 AoE).
- **Input Type**: free-text plus prior experimental evidence.
- **Reference Papers** (closest, to verify in Phase 2): ADC (CVPR 2024), LDC
  (ECCV 2024), EFC (ICLR 2024), AdaGauss (NeurIPS 2024), SDC (CVPR 2020),
  FeTrIL (WACV 2023), DeepInversion (CVPR 2020), ABD (ICCV 2021).
- **Compute**: JAX/Flax on Google TPUs; ResNet-18 on CIFAR-100 / TinyImageNet /
  ImageNet-Subset (warm- and cold-start protocols).
- **Kill gate**: on CIFAR-100 warm-start 10-step EFCIL, the full method must
  beat its own zero-noise and point-model ablations by at least 1 pp last-step
  accuracy over 3 seeds; otherwise the SB claim is dropped.

### Alternatives held for a possible pivot

- **B. Bridge-matching drift compensation**: transport stored class Gaussians
  with a stochastic map learned from paired old/new features of current data.
  Lower risk, lower novelty; "why SB and not heteroscedastic regression" is weak.
- **C. Sequential posterior SB sampler in LoRA/weight space**: half-bridge from
  the task t-1 posterior to posterior × likelihood, using the parent project's
  score networks, with cost measured in predictive space. Most novel, slowest,
  exposed to weight-symmetry problems.
