# Calibrated Test-Time Guidance for Bayesian Inference

[Project page](https://dgeyfman.github.io/calibrated-guidance-page/)

Reference implementation and full reproduction code for **Calibrated Test-Time
Guidance for Bayesian Inference** (Geyfman, Draxler, Groeneveld, Lee, Karaletsos,
Mandt; ICML 2026).

Common test-time guidance methods (DPS, LGD, …) use *biased* estimators of the
diffused likelihood and therefore do not sample the true Bayesian posterior. This
repo implements **Calibrated Bayesian Guidance (CBG)** — consistent gradient-free
and gradient-based estimators that converge to the correct posterior as compute
increases — and reproduces **every experiment in the paper from one place**, with
a single shared estimator and a uniform interface.

| Experiment | Paper | What CBG does | Our reproduction |
|---|---|---|---|
| **Bayesian-inference benchmark** | §6.1, Table 1 | gradient-free CBG, C2ST vs reference posteriors on 5 SBI tasks | avg C2ST **0.528** (paper 0.527), over 10 observations |
| **Black-hole imaging** | §6.2, Table 2 | gradient-free CBG, slow DDPM + fast mean-flow renoise | K=128/512/4096 → **25.1/26.0/26.3** dB (paper 25.4/26.1/26.4) |
| **4× super-resolution** | §6.3a, Table 5 | gradient-free CBG with a pixel mean-flow prior | renoise K=2048 → **~24.5** dB / **0.31** LPIPS (paper 24.41 / 0.298) |

> Black-hole PSNR has ~±0.3 dB run-to-run variance from the stochastic sampler;
> the paper number is the mean over draws (we verify this with a 10-seed run,
> §"Black-hole" below).

---

## 1. Repository walkthrough

```
calibrated_guidance/         ← THE shared estimator (the paper's contribution)
  guidance.py                  build_reinforce_mean_fn (Eq 20, gradient-free),
                               build_reparam_mean_fn (Eq 16, gradient-based)
  inference.py                 flow_matching / edm_sde samplers, mcmc
  diffusion_posterior/         p(x0|xt): analytic (gaussian, uniform, gmm) + base
  tasks/, mmd.py, utils.py
experiments/
  common/                      unified W&B logging + seeding (shared by all)
  sbi/                         §6.1 — 5 tasks, library posteriors + likelihoods, C2ST
  black_hole/                  §6.2 — clean driver over the vendored InverseBench engine
  super_resolution/            §6.3a — clean driver + the SR core (copied verbatim)
third_party/
  InverseBench/                vendored black-hole engine (verbatim; see UPSTREAM_DIFF.md)
  easy_meanflow/               modules needed to load the mean-flow checkpoint
  pMF/                         pixel-mean-flow code (fetched by download.sh, gitignored)
requirements/                  per-experiment dependency sets
slurm/                         one Slurm script per experiment + a smoke test
tests/                         comprehensive test suite (run in CI)
run.py                         uniform dispatcher: `python run.py <experiment> ...`
submit.sh + env.example.sh     cluster launcher + local-config template
```

The estimator in [`calibrated_guidance/`](calibrated_guidance/) is the single
source of truth: all three experiments import the *same* `build_reinforce_mean_fn`
/ `flow_matching` and the *same* analytic posteriors. Each experiment is a thin,
clean driver on top.

---

## 2. Install

The core library is pure PyTorch; each experiment adds its own dependencies.

```bash
git clone <this-repo> && cd Calibrated-Guidance
pip install -e ".[dev]"            # core library + test tooling

# then add the experiment(s) you want to run:
pip install -e ".[sbi]"            # scikit-learn (C2ST)
pip install -e ".[blackhole]"      # ehtim, hydra, ...
pip install -e ".[superres]"       # pMF deps (piq, fire, ...)
```

On the authors' cluster the `ibench` conda env already satisfies all three.

---

## 3. One-time setup: local paths + checkpoints

Private paths (checkpoints, datasets, your Slurm partition, your python) live in a
**gitignored** `env.local.sh`. Copy the template and fill it in:

```bash
cp env.example.sh env.local.sh
$EDITOR env.local.sh               # set PY, SLURM_PARTITION, *_CKPT, VAL_ROOT, ...
```

The Slurm scripts source it automatically, and [`submit.sh`](submit.sh) uses it to
launch jobs (`./submit.sh <experiment> [sbatch flags] ENV=val ...`). Nothing
private is ever committed.

**Fetch public assets** (per experiment; all paths are also overridable by flag):

```bash
bash experiments/black_hole/download.sh        # diffusion prior + test data (public InverseBench)
bash experiments/super_resolution/download.sh  # pMF code + pMF-B-16 checkpoint (HuggingFace Lyy0725/pMF)
# SBI: data is already in the repo (experiments/sbi/data/, 12 MB). Nothing to fetch.
```

| Asset | Source | Notes |
|---|---|---|
| black-hole diffusion prior + test set | public InverseBench (GitHub release + CaltechData) | `download.sh` |
| black-hole **mean-flow** ckpt | authors (upload pending) | set `MF_CKPT`; use the EMA `network-snapshot-*.pkl` |
| pixel mean-flow `pMF-B-16` | HuggingFace `Lyy0725/pMF` | `download.sh` |
| ImageNet-val (super-res) | standard ImageNet | set `VAL_ROOT` (+ optional `IMAGENET_VAL_TAR`) |
| SBI reference posteriors | committed (sbibm, Lueckmann et al.) | no action |

---

## 4. Reproduce the experiments

Every experiment runs through the same dispatcher (`python run.py <name> --help`)
and logs to one W&B project (`calibrated-guidance`, grouped by experiment). On a
cluster, wrap with `./submit.sh <name> ...` (it injects `--partition` + your env).

### 4a. Bayesian-inference benchmark (§6.1, Table 1) — CPU, minutes

The 5 SBI tasks with analytic priors. CBG draws 10⁴ posterior samples per task and
scores **C2ST** (↓, 0.5 = perfect) against the reference posteriors.

```bash
# paper config: N=100 outer steps, K=1000 candidates/step, gradient-free
python run.py sbi --num-steps 100 --num-particles 1000 --observations 1

# gradient-based estimator:
python run.py sbi --estimator reparam --num-steps 100 --num-particles 1000

# full Table 1 (average over all 10 observations, like the paper):
python run.py sbi --num-steps 100 --num-particles 1000 --observations 1-10
```

Prints a per-task C2ST table vs the paper targets. Averaged over all 10
observations (grad-free) we measure `0.505 / 0.515 / 0.582 / 0.509 / 0.528`,
average **0.528** (paper `0.505 / 0.513 / 0.584 / 0.507 / 0.525`, avg 0.527) —
each task within ~0.003. A single observation runs in ~1 min; a quick smoke:
add `--num-steps 30 --num-particles 200`.

### 4b. Black-hole imaging (§6.2, Table 2) — GPU

Two samplers for `p(x0|xt)`: `--posterior meanflow` (fast, one-step renoise — needs
the mean-flow checkpoint) and `--posterior ddpm` (slow DDPM inner loop). Defaults
match the paper (`stop_at_sigma=0.05`, `last_hop`, no clamp).

```bash
# single GPU, a handful of instances (after editing env.local.sh):
source env.local.sh
$PY run.py black-hole --posterior meanflow --reinf-k 512 --num-steps 100 \
    --id-list 0-9 --mf-ckpt "$MF_CKPT" --data-dir "$BH_DATA" --prior-ckpt "$BH_PRIOR"

# full 100-task reproduction, sharded across the cluster (Table 2 rows):
REINF_K=128 POSTERIOR=meanflow ./submit.sh black_hole --array=0-19   # ~25.4 dB
REINF_K=512 POSTERIOR=meanflow ./submit.sh black_hole --array=0-19   # ~26.1 dB
REINF_K=4096 POSTERIOR=meanflow CHUNK=2 ./submit.sh black_hole --array=0-49  # ~26.4 dB
```

Aggregate the per-image PSNR from `slurm_logs/bh_<jobid>_*.out`. The
result is **mean over the 100 tasks**; individual runs vary ±~0.3 dB. To see this,
run 10 seeds (`SEED=0 … SEED=9`) — their means cluster on the paper number
(we measured mean **25.99**, range **[25.84, 26.13]** for K=512 vs paper 26.10).

### 4c. 4× super-resolution (§6.3a, Table 5) — needs a ≥40 GB GPU

The paper config: **renoise** sampler, **K=2048** candidates, **8 outer steps**,
σ_y=0.01, cfg ω=3.0, 4096 tiles, on the 1000-image SR3 list with pMF conditioned
on the classifier's top-1 prediction. K=2048 needs a 48 GB GPU (the paper split it
across 4× Titan RTX; one 48 GB GPU holds the whole batch, ~73 s/image).

```bash
# 1-image smoke (set VAL_ROOT + SR_CKPT in env.local.sh first):
source env.local.sh
$PY run.py super-resolution --limit 1 --num-particles 2048 --num-outer-steps 8 \
    --cfg-omega 3.0 --noise-std 0.01 \
    --sr3-top1k-path experiments/super_resolution/sr3/sr3_top1k.txt \
    --predicted-labels-csv experiments/super_resolution/sr3/sr3_top1k_predicted_labels.csv \
    --val-root "$VAL_ROOT" --checkpoint "$SR_CKPT"

# full 1000-image SR3 run, sharded on 48 GB nodes:
./submit.sh super_resolution --array=0-19 --nodelist=<48GB-nodes> --mem=80G
```

PSNR / LPIPS are written per image to `--output-dir/per_image_metrics.csv`.
Reproduces **~24.5 dB / 0.31 LPIPS** (paper 24.41 / 0.298).

### Quick end-to-end check (all three, minutes)

```bash
./submit.sh smoke         # tiny configs for SBI + black-hole + super-resolution
```

---

## 5. Tests

A comprehensive suite covers the estimator math, the analytic posteriors, the
samplers, and each experiment (the SBI tasks reproduce the *exact* analytic
Gaussian posterior end-to-end).

```bash
pip install -e ".[dev]"
pytest                                                            # everything
pytest -m "not gpu and not blackhole and not superres and not slow"   # what CI runs (CPU, ~4 min)
```

CI runs the CPU suite on every push across Python 3.10–3.12
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

---

## 6. Attribution

This repo builds on public work, vendored or fetched with attribution:

- **InverseBench** — black-hole benchmark ([devzhk/InverseBench](https://github.com/devzhk/InverseBench), ICLR 2025). Our fork adds the CBG estimators; see [`third_party/InverseBench/UPSTREAM_DIFF.md`](third_party/InverseBench/UPSTREAM_DIFF.md).
- **Pixel Mean Flows (pMF)** — one-step ImageNet prior ([Lyy-iiis/pmf](https://github.com/Lyy-iiis/pmf), arXiv 2601.22158).
- **sbibm** — SBI benchmark tasks + reference posteriors ([Lueckmann et al. 2021](https://github.com/sbi-benchmark/sbibm)).
- **easy_meanflow** — trains the black-hole one-step mean-flow model.

## 7. Citation

```bibtex
@inproceedings{geyfman2026calibrated,
  title     = {Calibrated Test-Time Guidance for Bayesian Inference},
  author    = {Geyfman, Daniel and Draxler, Felix and Groeneveld, Jan and
               Lee, Hyunsoo and Karaletsos, Theofanis and Mandt, Stephan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
