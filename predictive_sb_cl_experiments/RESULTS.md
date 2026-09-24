# Initial falsification results

These experiments were run on two NVIDIA Titan RTX GPUs. They are intended to
answer the narrow first question: does an empirical predictive Schrödinger
bridge improve over direct distillation to the same functional-posterior
endpoint under the same neural optimisation budget?

## Five-task Permuted MNIST

Configuration: eight particles, 256 inducing inputs, 64 current probes, 1,200
first-task updates, and 600 updates at every later boundary. The main direct and
best-SB comparison uses three seeds.

| Method | Seeds | Average accuracy | Forgetting | Mean NLL | Mean ECE |
|---|---:|---:|---:|---:|---:|
| Direct endpoint distillation | 3 | 0.9490 | 0.0243 | 0.2335 | 0.0867 |
| Predictive SB, best lifting | 3 | 0.9482 | 0.0253 | **0.2301** | **0.0828** |
| Deterministic OT path | 1 | 0.9491 | 0.0234 | 0.2311 | 0.0866 |
| Random coupling path | 1 | 0.9419 | 0.0321 | 0.2692 | 0.1034 |
| Fine-tuning | 1 | 0.8354 | 0.1689 | 0.5139 | 0.0618 |

The predictive representation is effective, and functional endpoint coupling
matters: random coupling is substantially worse. The stochastic bridge does not
improve accuracy over direct distillation, although it modestly improves NLL and
ECE in this short stream.

## Ten-task Permuted MNIST

With only 256 inducing inputs, both direct and SB variants fall to roughly
86.8% average accuracy. Increasing the fixed memory to 1,024 inputs and using
the zero-temperature functional-posterior limit gives:

| Method | Seeds | Average accuracy | Forgetting | Mean NLL | Mean ECE |
|---|---:|---:|---:|---:|---:|
| Direct endpoint distillation | 3 | **0.9062** | **0.0659** | **0.4556** | **0.1739** |
| Predictive SB, best lifting | 3 | 0.9055 | 0.0660 | 0.4606 | 0.1771 |

The methods are effectively tied in accuracy and forgetting. The five-task
calibration advantage does not persist over ten tasks.

## Bridge diagnostics

- Sinkhorn marginal error is typically below `2e-7`.
- Stochastic trajectories hit their assigned terminal logits within about
  `4e-6` RMS.
- A Birkhoff--von Neumann decomposition samples a permutation-valued coupling,
  preserving both finite empirical endpoint marginals for every lifted
  population.
- Exact per-PC whitening was rejected after a diagnostic run showed that it
  maps the maximal-rank empirical endpoints to a regular simplex and makes the
  coupling uniform. The implementation uses a global PCA scale and preserves
  relative predictive geometry.
- Raising entropic regularisation by 10x reduced five-task accuracy to 0.8838;
  the useful regime is close to the low-entropy/optimal-transport limit.

## Conclusion

The implementation validates that a genuine empirical SB can be constructed
and lifted in predictive space. It does **not** validate the stronger research
hypothesis that stochastic SB paths improve continual-learning accuracy. The
current performance comes primarily from inducing-point functional
regularisation; direct endpoint distillation is the stronger accuracy baseline.

An ICLR paper should therefore not yet use the bridge as its central empirical
claim. A defensible next hypothesis is to learn a reference process aligned
with neural optimisation, instead of using Brownian motion in predictive PCA
coordinates. Any such method must still beat the direct endpoint control used
here.
