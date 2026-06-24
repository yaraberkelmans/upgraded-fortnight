# Riot model Sobol pilot (N=64)

This folder runs a first-order Sobol pilot with exactly **384 model runs**:

- 4 varied parameters
- base sample size `N=64`
- `calc_second_order=False`
- number of rows: `64 * (4 + 2) = 384`
- one fixed seed (`42`) for every Sobol row

## Varied parameter ranges

| Parameter | Minimum | Maximum |
|---|---:|---:|
| `similarity_threshold` | 0.05 | 0.90 |
| `fight_threshold` | -0.35 | 0.45 |
| `hawk_dove_C` | 0.10 | 19.00 |
| `police_density` | 0.01 | 0.15 |

Fixed choices include `home_fraction=0.5`, `hawk_dove_strategy=logit_prior`, `logit_beta=5.0`, 100 riot burn-in steps, and 100 measurement steps.

## Folder layout

- `model/`: unchanged model files
- `run_sobol.py`: creates samples and executes 384 runs
- `analyze_sobol.py`: calculates S1 indices and creates plots
- `run_all.slurm`: Snellius job script
- `data/`: generated NumPy arrays and Sobol tables
- `plots/`: generated PNG plots
- `logs/`: Slurm logs

## Run on Snellius

From this folder:

```bash
sbatch run_all.slurm
```

Follow the job:

```bash
squeue -u $USER
```

## Generated data

- `data/sobol_samples.npy`: shape `(384, 4)`
- `data/run_results.npy`: 384 structured rows
- `data/sobol_outputs.npz`: output arrays
- `data/timing.json`: runtime information
- `data/sobol_first_order.csv`
- `data/sobol_first_order.json`
- `plots/sobol_S1_*.png`
- `plots/sobol_S1_overview.png`

If any run fails or does not finish spatial warm-up within 2,000 steps, it is recorded in `data/invalid_runs.npy`. The analysis script then stops instead of silently producing invalid Sobol indices.

## Per-run meetbestanden

Elke geldige run schrijft nu een eigen bestand naar:

```text
data/runs/run_0000.npy
data/runs/run_0001.npy
...
data/runs/run_0383.npy
```

Elk bestand bevat exact 100 records, een voor iedere meetstap na de ruimtelijke warm-up en de vaste riot burn-in van 100 stappen. Open bijvoorbeeld met:

```python
import numpy as np

run = np.load("data/runs/run_0000.npy", allow_pickle=False)
print(run.shape)
print(run.dtype.names)
print(run["fighting"])
```

De compacte `data/run_results.npy` blijft bestaan en wordt gebruikt voor de Sobol-analyse. Ongeldige runs krijgen geen meetbestand; hun foutcode staat in `invalid_runs.npy`.
