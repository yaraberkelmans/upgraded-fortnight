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
    riot_params=RiotParams(police_density=0.05, hawk_dove_strategy="aggressiveness"),
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

1. **Warmup** — Schelling movement only, no fighting. Ends when the CV (std/mean) of spatial entropy over the last `warmup_window` steps drops below `warmup_cv_threshold`, or when no fan moved.
2. **Main simulation** — movement and fighting both run. Starts automatically after warmup converges.

### How a fan decides each step

1. **Happiness / segregation move** — a fan counts same-group neighbours within radius 1. If the fraction falls below `similarity_threshold` the fan moves to a nearby empty cell. Even happy fans have a `random_move_chance` probability of moving anyway, keeping the grid from freezing completely. Movement destination is weighted by `exp(-movement_decay × distance)`, preferring nearby empty cells.

2. **Perceived probabilities** — within `fan_vision` cells the fan counts friends, enemies, and police and computes:
   - `perceived_win_probability = exp(-k × enemies/friends)`
   - `perceived_arrest_probability = 1 − exp(-k × 5·police/total_fans)`

3. **Fight decision** — `fight_want = aggressiveness × perceived_win_probability`. If `fight_want − perceived_arrest_probability > fight_threshold` the fan picks an opponent and plays a Hawk-Dove round; "hawk" means fighting.

### Hawk-Dove strategies

| Strategy | Behaviour |
|---|---|
| `logit` **(default)** | Play hawk with probability `1 / (1 + exp(-logit_beta × fight_want))`. Smooth, continuous response to how much a fan wants to fight. |
| `aggressiveness` | Play hawk with probability equal to the fan's own aggressiveness (drawn from a Beta distribution at spawn). |
| `nash_ess` | Play hawk with probability `V/C` where V is the number of visible friends and C is the injury cost parameter. |
| `bourgeois` | Home fans always play hawk; away fans always play dove. |
| `anti_bourgeois` | Away fans always play hawk; home fans always play dove. |
| `tit_for_tat` | Mirror what the opponent played in their last encounter. |

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
| `zone_size` | 10 | Side length of zones used when computing spatial entropy. |
| `warmup_cv_threshold` | 0.01 | Warmup ends when the CV (std/mean) of entropy over the last `warmup_window` steps falls below this. |
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
| `hawk_dove_strategy` | `logit` | Strategy used by all fans when playing Hawk-Dove (see table above). |
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
- `Spatial entropy` — Shannon entropy of group mixing across zones (0 = fully segregated, ln 2 ≈ 0.69 = fully mixed).
- `Entropy CV` — coefficient of variation of entropy over the last `warmup_window` steps; used to detect warmup stabilisation.
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
server_riot_model.py            # Solara visualisation server
requirements.txt
```
