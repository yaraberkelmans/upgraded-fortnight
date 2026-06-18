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

### How a fan decides each step

1. **Happiness / segregation move** — a fan counts same-group neighbours within radius 1. If the fraction falls below `similarity_threshold` the fan moves to a nearby empty cell, preferring closer cells (weighted by `exp(-movement_decay × distance)`).

2. **Perceived probabilities** — within `fan_vision` cells the fan counts friends, enemies, and police and computes:
   - `perceived_win_probability = exp(-k × enemies/friends)`
   - `perceived_arrest_probability = 1 − exp(-k × 5·police/total_fans)`

3. **Fight decision** — `fight_want = aggressiveness × perceived_win_probability`. If `fight_want − perceived_arrest_probability > fight_threshold` the fan picks an opponent and plays a Hawk-Dove round; "hawk" means fighting.

### Hawk-Dove strategies

| Strategy | Behaviour |
|---|---|
| `aggressiveness` | Play hawk with probability equal to the fan's own aggressiveness (drawn from a Beta distribution at spawn). |
| `nash_ess` | Play hawk with probability `V/C` where V is the number of visible friends and C is the injury cost parameter. |
| `bourgeois` | Home fans always play hawk; away fans always play dove. |
| `anti_bourgeois` | Away fans always play hawk; home fans always play dove. |
| `tit_for_tat` | Mirror the opponent's last play. |

### Parameters

#### Segregation (`SegregationParams`)

| Parameter | Default | Description |
|---|---|---|
| `N` | 40 | Grid side length (N × N cells). |
| `agent_density` | 0.80 | Fraction of cells occupied by fans. |
| `home_fraction` | 0.50 | Fraction of fans that are home supporters. |
| `similarity_threshold` | 0.30 | Minimum fraction of same-group neighbours for a fan to be "happy". |
| `movement_decay` | 1.0 | Controls how strongly fans prefer nearby empty cells when moving. |
| `steps` | 100 | Default number of steps for `run_model()`. |
| `seed` | 42 | Random seed. |
| `torus` | True | Wrap edges (toroidal grid). |
| `count_empty_as_different` | True | Count empty cells as dissimilar neighbours (lecture-style Schelling). |

#### Riot (`RiotParams`)

| Parameter | Default | Description |
|---|---|---|
| `police_density` | 0.05 | Fraction of cells occupied by police. |
| `perception_k` | 0.693 | Sensitivity constant in the win/arrest probability formulae (≈ ln 2). |
| `fan_vision` | 2 | Chebyshev radius within which a fan perceives others. |
| `fight_threshold` | 0.0 | Minimum value of `fight_want − P(arrest)` required to start a fight. |
| `police_vision` | 5 | Chebyshev radius within which police can spot fighting fans. |
| `hawk_dove_strategy` | `aggressiveness` | Strategy used by all fans when playing Hawk-Dove (see table above). |
| `hawk_dove_C` | 4.0 | Injury cost parameter used by the `nash_ess` strategy. |

### Collected metrics

The `DataCollector` records these model-level variables every step:

- `Happy` / `Unhappy` — number of fans at or below the similarity threshold.
- `Home` / `Away` — fan counts per group (decreases as arrests accumulate).
- `Average similarity` / `Segregation index` — mean same-fraction across all fans.
- `Moves` — total fan moves this step.
- `Average last move distance` / `Average last move distance (moved fans)`.
- `Police` — number of police still on the grid.
- `Fighting fans` — fans playing "hawk" this step.
- `Arrests this step`.
- `Average aggressiveness`, `Average perceived win probability`, `Average perceived arrest probability`.

### Project structure

```
riot_model/
    __init__.py
    riot_model.py       # Fan, Police, RiotModel, SegregationParams, RiotParams
server_riot_model.py    # Solara visualisation server
requirements.txt
```
