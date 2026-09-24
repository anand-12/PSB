# psb_tpu: posterior-bridge continual-learning sweeps on Cloud TPU

A JAX port of the `bwd_cl` experiments that packs many small-MLP runs into one
program (runs × particles, vmapped and sharded across TPU chips). Upload the
zip, run two scripts, collect tables.

## Quick start

**One-time GCP setup** (from the TPU Builders guide):
1. Use a **brand-new** GCP project linked to the credit billing account.
2. In IAM & Admin → Quotas, check that `GPUS_ALL_REGIONS` is not 0, and request
   8 chips of `PREEMPTIBLE_TPU_V6E` in the region of `ZONE`. Mention the TPU
   Builders Program; approval can take up to 2 business days.
3. `gcloud auth login && gcloud config set project <id> && gcloud services enable tpu.googleapis.com`
4. **Budget alert:** Billing → Budgets & alerts → $15,000 with alerts at 50/75/90%.
   Alerts only email you; they do not stop spending. The guardrails below do.

**Every time** (from your laptop, in the unzipped `psb_tpu/` folder):
```bash
nano cloud/config.env                              # set PROJECT; keep the other defaults
bash cloud/create_disk.sh                          # first time only
bash cloud/create_vm.sh                            # Flex-start v6e-4; waits for capacity
bash cloud/upload_and_run.sh ../psb_tpu.zip        # setup + start the sweep in tmux
bash cloud/status.sh                               # progress
bash cloud/fetch_results.sh                        # tables -> ./results_tpu
bash cloud/teardown.sh                             # if SELF_DELETE=0
```
`upload_and_run.sh` runs `setup.sh` on the VM: uv, Python 3.12 (Ubuntu 22.04's
3.10 is too old for JAX 0.10), `jax[tpu]==0.10.2` and the unit tests. It then
starts `run_all.sh` in a tmux session named `psb`. To work on the VM directly:
`gcloud compute ssh psb-tpu --zone us-east5-a`, then `tmux attach -t psb`.

## Guardrails against runaway cost
| Guard | Where | Default |
|---|---|---|
| VM deletes itself after a fixed time | `MAX_RUN` in `config.env` | 24 h (worst case ≈ $130 on v6e-4) |
| Sweep stops at an estimated spend | `COST_CAP_USD` | $150 |
| VM deleted when the sweep ends | `SELF_DELETE=1` | on |
| Results survive VM deletion | on the `psb-data` disk (+ optional GCS) | on |

Flex-start VMs cannot be stopped, only deleted, so always delete when done.

## What runs (in order; `PHASES=...` selects a subset)
| Phase | Question | Runs | Base-run eq.* |
|---|---|---|---|
| `smoke` | every code path works (≈1 min) | 12 | ~0 |
| `phase0_parity` | port reproduces PyTorch; **gate**, stops if it fails | 14 | 7 |
| `phase1a_tune` | tune EWC / SI / finetune in our code | 114 | 57 |
| `phase1b_seeds` | 20-seed comparison, paired vs PSB | 160 | 120 |
| `phase2a_schedules` | does any β schedule beat β = 1? | 110 | 110 |
| `phase2b_budget` | corrected Fig. 2: accuracy vs moves, arms matched | 75 | 419 |
| `phase3_streams` | 20/50/100 tasks × Fisher decay γ | 117 | 652 |
| `phase4_backlog` | Rotated/Split λ ≥ 16, K = 1…1024, Laplace inits | 160 | 214 |

\*1 base-run equivalent = one 10-task Permuted MNIST run with K = 128 and
2,500 moves per boundary. Measured locally: ≈60 per TITAN RTX GPU-hour when
packed, so the full plan (~1,580) is ≈26 GPU-hours. On a v6e-4 I expect
roughly 2–5 hours, i.e. $10–30 at $5.40/h; this is an estimate. Phase 0's
timing on the TPU calibrates it.

## Read this before interpreting the results
Local validation (2× TITAN RTX, 5 seeds unless noted) reproduces PyTorch:
PSB K=128 **0.9119 ± 0.0013** (PyTorch 0.9099 ± 0.0013). In the Fig. 2 setup
(K=32, batch 128, 3 seeds): PSB 0.9040 (PyTorch 0.9011), and the "β = 1" arm
0.8953 (PyTorch 0.8890).

That "β = 1" arm in PyTorch **also used a cosine learning-rate decay**; the
PSB arm did not. With everything else matched (β = 1, constant LR), β = 1
scores **0.9225 ± 0.0020 vs PSB's 0.9119** at K=128 (paired +1.07 ± 0.13).
In the Fig. 2 setup it scores 0.9148 vs 0.9040. So at 2,500 moves the
tempered path currently costs about a point, and Fig. 2's gap came from the
LR schedule. `phase1b`, `phase2a` and `phase2b` carry the matched `beta1` /
`constant` arm throughout, to settle this at scale and across budgets.

## Outputs (on the VM: `/mnt/data/psb_results/<phase>/`)
- `<tag>_seed<s>/summary.json`: config, final metrics, full accuracy matrix, per-task history.
  Same schema as the PyTorch runs, so `scripts/make_figures.py` can read them.
- `<tag>_seed<s>/metrics.jsonl`: one line per task (also β schedule and per-stage loss variance).
- `summary.md` / `summary.csv`: mean ± sd per tag, paired difference vs the grid's `compare_to`.
- `logs/`: one log per phase plus `run_all.log`.

## Methods and knobs (`psb/config.py`)
- `method`: `psb` (tempered data step + exact proximal Fisher prior), `ewc`
  (online EWC through Adam), `si` (Synaptic Intelligence), `finetune`.
- `schedule`: `linear`, `power` (`schedule_param` = p), `cosine`, `hold`
  (`schedule_param` = held fraction), `constant` (β = 1), `thermo`
  (equal thermodynamic length).
- `lr_decay`: 0 = constant LR (PSB default); 1 = cosine to zero over each boundary.
- `init`: `independent` (K trained nets) or `laplace` (samples around
  `init_modes` trained nets, width `laplace_temperature`).
- Numeric knobs (`lam`, `gamma`, `lr`, `lr_decay`, `schedule_param`, …) are
  vmapped, so a sweep over them costs one compilation. Shape knobs (`particles`,
  `tasks`, `stages`, `moves_per_stage`, `method`, …) define compilation groups.

A new experiment is a new YAML in `grids/` (see `grids/smoke.yaml`). Preview it with
`python -m psb.run grids/x.yaml --plan`, run it with `python -m psb.run grids/x.yaml`,
and tabulate with `python -m psb.summarize /mnt/data/psb_results/x`.

## Troubleshooting (from the TPU Builders guide)
- Stuck in `WAITING_FOR_RESOURCES` / `STOCKOUT`: usually `GPUS_ALL_REGIONS = 0`.
  Link a card or bank account to the billing profile for identity verification,
  then request 8–16 in Quotas. Or try another v6e zone.
- `code 10, tenant project creation`: create a new project.
- Only **single-host** machines are supported (`ct6e-standard-1t`, `-4t`, `-8t`);
  multi-host slices would need `jax.distributed`.
- Preempted or killed: rerun `bash run_all.sh` (or `upload_and_run.sh`). It
  skips finished runs and resumes the rest from the last checkpoint.
- Support: tpu-builders-support@google.com (include your project ID).

## Layout
```
psb/        config, data, model, core (methods), schedules, engine, run, summarize, gate
grids/      smoke + phase0 ... phase4
tests/      unit tests (run on CPU: JAX_PLATFORMS=cpu pytest -q tests)
cloud/      laptop-side gcloud scripts + config.env
data/       mnist.npz (bundled, 11 MB)
setup.sh    VM environment;  run_all.sh  the whole sweep
```
