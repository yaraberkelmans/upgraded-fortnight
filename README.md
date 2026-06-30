# ABM - Group 5
Agent Based Modelling assignment — a spatial simulation of football fan segregation and riot dynamics built on [Mesa](https://mesa.readthedocs.io/).

---

## Getting started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the interactive visualisation

```bash
solara run server_riot_model.py
```

This opens a browser-based dashboard with a live grid, parameter sliders, and several charts.

### Run the model headlessly (no UI)

```python
from riot_model.riot_model import RiotModel, SegregationParams, RiotParams

model = RiotModel(
    segregation_params=SegregationParams(N=40, steps=200, seed=42),
    riot_params=RiotParams(police_density=0.05, hawk_dove_strategy="logit_prior"),
)
model.run_model()

data = model.datacollector.get_model_vars_dataframe()
print(data.tail())
```

---

## The Riot Model

The model simulates two groups of football fans (home and away) moving around an N×N grid. It combines a **Schelling segregation** layer with a **Hawk-Dove** fight-decision layer and a **police** arrest layer.

### Agents

| Agent | Description |
|---|---|
| `Fan` | Moves when unhappy with its neighbourhood; decides whether to fight nearby opponents each step. |
| `Police` | Patrols the grid; moves toward the nearest fighting fan and arrests them. |

### Simulation phases

The model has two phases, controlled by an `in_warmup` flag checked inside every `step()` call:

1. **Warmup** — Schelling movement only, no fighting. Ends when the CV (std/mean) of fine-grained spatial entropy (`zone_size_fine`) over the last `warmup_window` steps drops below `warmup_cv_threshold`, or when no fan moved.
2. **Main simulation** — movement and fighting both run. Starts automatically after warmup converges.

### How a fan decides each step

1. **Happiness / segregation move** — a fan counts same-group neighbours within radius 1. If the fraction falls below `similarity_threshold` the fan moves to a nearby empty cell. Even happy fans have a `random_move_chance` probability of moving anyway, keeping the grid from freezing completely. The destination is sampled from the empty cells within a radius-5 window, weighted by `exp(-movement_decay × distance)` (with a fall-back to a global search only if no empty cell is within range), preferring nearby cells.

2. **Perceived probabilities** — within `fan_vision` cells the fan counts friends, enemies, and police and computes:
   - `perceived_win_probability = exp(-k × enemies/friends)`
   - `perceived_arrest_probability = 1 − exp(-k × 5·police/total_fans)`

3. **Fight decision** — `fight_want = aggressiveness × perceived_win_probability`. If `fight_want − perceived_arrest_probability > fight_threshold` the fan picks an opponent and plays a Hawk-Dove round; "hawk" means fighting.

### Hawk-Dove strategies

All strategies operate on the standard symmetric Hawk-Dove payoff matrix: two hawks split `(V − C)/2`; hawk vs dove gives `V` to the hawk; two doves split `V/2`. `V` is the number of same-team neighbours within `fan_vision`; `C` is `hawk_dove_C`.

| Strategy | Behaviour |
|---|---|
| `nash_ess` | Mixed Nash / ESS: play hawk with probability `min(1, V/C)`. |
| `logit_prior` **(default)** | Logit QRE with a uniform prior: assumes the opponent plays hawk with probability 0.5. Expected payoff difference `ΔE = V/2 − C/4`; hawk probability `1 / (1 + exp(−logit_beta × ΔE))`. |
| `logit_qre` | Logit QRE with an empirical prior: opponent hawk probability `q` is estimated from the local enemy/friend ratio. `ΔE = V/2 − q·C/2`; hawk probability `1 / (1 + exp(−logit_beta × ΔE))`. |

### Parameters

#### Segregation (`SegregationParams`)

| Parameter | Default | Description |
|---|---|---|
| `N` | 40 | Grid side length (N × N cells). |
| `agent_density` | 0.80 | Fraction of cells occupied by fans. |
| `home_fraction` | 0.50 | Fraction of fans that are home supporters. |
| `similarity_threshold` | 0.30 | Minimum fraction of same-group neighbours for a fan to be "happy". |
| `movement_decay` | 1.0 | Controls how strongly fans prefer nearby empty cells when moving. |
| `steps` | 100 | Default number of steps for `run_model()` and max warmup steps. |
| `seed` | 42 | Random seed. |
| `torus` | True | Wrap edges (toroidal grid). |
| `count_empty_as_different` | True | Count empty cells as dissimilar neighbours (lecture-style Schelling). |
| `zone_size` | 10 | Side length of coarse zones. Unused — only fine-grained entropy (`zone_size_fine`) is computed. |
| `zone_size_fine` | 4 | Side length of fine zones used when computing fine-grained spatial entropy (also used for warmup convergence). |
| `warmup_cv_threshold` | 0.01 | Warmup ends when the CV (std/mean) of fine-grained entropy over the last `warmup_window` steps falls below this. |
| `warmup_window` | 10 | Rolling window size for the CV stabilisation check. |
| `random_move_chance` | 0.005 | Probability a happy fan moves anyway; prevents the grid from fully freezing. |

#### Riot (`RiotParams`)

| Parameter | Default | Description |
|---|---|---|
| `police_density` | 0.05 | Fraction of cells occupied by police. |
| `perception_k` | 0.693 | Sensitivity constant in the win/arrest probability formulae (≈ ln 2). |
| `fan_vision` | 2 | Chebyshev radius within which a fan perceives others. |
| `fight_threshold` | 0.0 | Minimum value of `fight_want − P(arrest)` required to start a fight. |
| `police_vision` | 5 | Chebyshev radius within which police can spot fighting fans. |
| `hawk_dove_strategy` | `logit_prior` | Strategy used by all fans when playing Hawk-Dove (see table above). |
| `hawk_dove_C` | 4.0 | Injury cost parameter used by the `nash_ess` strategy. |
| `logit_beta` | 5.0 | Steepness of the logit hawk-probability curve; higher = sharper threshold. |
| `aggressiveness_mean` | `None` | If set, all fans use this aggressiveness mean; if `None`, home fans use `home_fraction` and away fans use `1 − home_fraction`. |
| `aggressiveness_concentration` | 12.0 | Concentration of the Beta distribution for aggressiveness; higher = tighter spread. |
| `fighting_enabled` | `True` | When `False`, fans never fight and police never arrest (warmup-only mode). |

### Collected metrics

The `DataCollector` records these model-level variables every step:

- `Happy` / `Unhappy` — number of fans at or above/below the similarity threshold.
- `Home` / `Away` — fan counts per group.
- `Average similarity` / `Segregation index` — mean same-fraction across all fans.
- `Moves` — total fan moves this step.
- `Average last move distance` / `Average last move distance (moved fans)`.
- `Police` — number of police on the grid.
- `Fighting fans` — fans playing "hawk" this step.
- `Arrests this step` — arrests in the current step.
- `Total arrests` — cumulative arrests since the simulation started.
- `Spatial entropy (fine)` — Shannon entropy of group mixing across fine zones (`zone_size_fine`); 0 = fully segregated, ln 2 ≈ 0.69 = fully mixed.
- `Entropy CV (fine)` — coefficient of variation of fine-grained spatial entropy over the last `warmup_window` steps; this is the signal used to end warmup.
- `In warmup` — 1 during warmup phase, 0 once the main simulation begins.
- `Average aggressiveness`, `Average perceived win probability`, `Average perceived arrest probability`.

### Fan replacement on arrest

When a fan is arrested the population size stays constant: a new fan of a randomly chosen group is spawned at a random empty cell. This means `Home` and `Away` counts can drift over time as arrests accumulate, but total fan count stays stable.

### Project structure

```
riot_model/
    __init__.py
    fan.py                      # Fan agent, FanGroup, HawkDoveStrategy
    police.py                   # Police agent
    riot_model.py               # RiotModel, SegregationParams, RiotParams
    run_aggression_and_plot.py  # Headless batch runner / plotting helper
riot_model_refactor/            # Vectorized variant — see section below
    __init__.py
    fan.py
    police.py
    riot_model.py
server_riot_model.py            # Solara visualisation server (uses riot_model)
compare_refactor.py             # Equivalence + speed comparison of both versions
requirements.txt
```

---

## Sensitivity analysis data

---

## Two implementations

The model ships in two packages with the **same** public API (`RiotModel`,
`SegregationParams`, `RiotParams`) and the **same** datacollector columns:

| Package | Use it for |
|---|---|
| `riot_model` | The readable reference implementation. Drives the Solara server. |
| `riot_model_refactor` | A vectorized variant for batch work — parameter sweeps and sensitivity analysis. |

The two produce the same results on a given seed; `riot_model_refactor` only
differs in how it does its per-step bookkeeping.

```python
from riot_model_refactor.riot_model import RiotModel, SegregationParams, RiotParams

model = RiotModel(SegregationParams(N=40, steps=200, seed=42), RiotParams())
model.run_model()
```

### How `riot_model_refactor` is built

Movement and fighting are agent-driven (they are sequential and consume the RNG
in a fixed order). The per-step counting and reporting are spatial instead of
per-agent — rebuilt once per tick as numpy arrays:

- **Spatial occupancy planes.** `_build_spatial_state()` builds three `(N, N)`
  integer planes (`home`, `away`, `police`) from current agent positions in a
  single pass over the agents.
- **Vectorized neighbour counts.** Friend/enemy/police counts (radius
  `fan_vision`) and same-group counts (radius 1) are computed for every cell at
  once with a torus-aware box-sum (`_box_sum`, pure numpy), rather than a
  per-fan `grid.get_neighbors(...)` call.
- **Cached aggregates.** The datacollector reporters (`count_happy`,
  `average_aggressiveness`, …) read scalars accumulated during the rebuild.
- **Gated fighting.** A fan only issues a `get_neighbors` call (needed to pick a
  random opponent) when the spatial planes show an adjacent opponent **and** its
  fight margin clears `fight_threshold`.

### Verifying equivalence and speed

`compare_refactor.py` runs both packages across several seeds and both torus
settings, asserts that the integer/count columns are exactly equal and the
float-average columns match within tolerance, and reports the speedup:

```bash
python compare_refactor.py
```

`riot_model_refactor` is roughly **1.4× faster** than `riot_model` on default
parameters; the remaining runtime is mostly the sequential movement and police
logic. As a rough planning figure, a 200-step run at default parameters takes on
the order of ~15 s, so an embarrassingly-parallel 3000-run sensitivity analysis
fits within ~1 hour on a 24-core node (more if the sweep pushes `N` or the vision
radii higher).

---

## Sensitivity analysis data

The `data/` directory is **not tracked in git** — download it separately and place it at the project root before running any analysis scripts.

It contains output from a Sobol sensitivity analysis: **5 random seeds × 3,072 Saltelli-sampled runs = 15,360 runs** total. Each seed folder uses an identical Sobol design (same `sobol_sampling_seed = 2026`), so seed-to-seed variation reflects pure stochasticity.

```
data/
  seed_43/          # repeated for seeds 44–47
    metadata.json           # sweep config: parameter bounds, fixed params, worker count
    sobol_samples.npy       # (3072, 4)    raw Sobol samples in [0,1]^4
    run_results.npy         # (3072,)      one row per run — parameters + aggregated outputs
    run_overview.npy        # (3072,)      entropy at key time-points, warmup duration
    runs/
      run_XXXX.npy          # (100,)       per-step time series for the measurement window
    arrests/
      arrests_XXXX.npy      # (N_arrests,) microdata — one row per arrest event
    overview/
      overview_XXXX.npy     # (1,)         per-run mirror of run_overview (convenience)
```

### File schemas

**`run_results.npy`** — aggregated outputs, one row per run

| Field | Description |
|---|---|
| `sample_id`, `seed`, `valid`, `failure_code` | Run metadata |
| `similarity_threshold`, `fight_threshold`, `hawk_dove_C`, `police_density` | Sobol input parameters |
| `warmup_steps`, `runtime_seconds` | Timing |
| `mean_fighting`, `std_fighting`, `peak_fighting`, `mean_fighting_fraction` | Fighting activity during measurement |
| `arrests_measurement`, `mean_arrests_per_step` | Arrest totals during measurement |
| `mean_spatial_entropy_local` | Mean fine-grained spatial entropy during measurement |
| `mean_similarity`, `mean_happy_fraction` | Schelling satisfaction during measurement |
| `mean_win_probability`, `mean_arrest_probability` | Perceived probabilities during measurement |

**`run_overview.npy`** — entropy at phase boundaries

| Field | Description |
|---|---|
| `run_id`, `valid` | Run metadata |
| `warmup_steps` | Steps until Schelling convergence |
| `warmup_entropy` | Spatial entropy at end of warmup |
| `start_measurement_entropy` | Spatial entropy after 100-step burn-in |
| `end_measurement_entropy` | Spatial entropy at end of measurement |

**`runs/run_XXXX.npy`** — per-step time series (100 steps, measurement window only)

`step`, `fighting`, `home`, `away`, `police`, `happy`, `unhappy`, `moves`, `arrests_step`, `measurement_total_arrests`, `spatial_entropy_local`, `average_similarity`, `happy_fraction`, `fighting_fraction`, `average_win_probability`, `average_arrest_probability`

**`arrests/arrests_XXXX.npy`** — arrest microdata (variable length, one row per arrest)

`step`, `is_respawn`, `aggressiveness`, `is_home`

> `is_respawn = True` means the arrested fan was itself a replacement spawned by a prior arrest. The true time-to-arrest for respawned fans is not recoverable from saved data (spawn step is not recorded).
