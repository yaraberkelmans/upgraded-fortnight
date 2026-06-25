# Riot model — arrest events and entropy overview

Deze versie bewaart de bestaande 100 measurement snapshots en voegt twee multiprocessing-veilige bestanden per run toe.

## Outputstructuur

```text
data/
├── runs/
│   └── run_XXXX.npy
├── arrests/
│   └── arrests_XXXX.npy
├── overview/
│   └── overview_XXXX.npy
├── run_overview.npy
├── run_results.npy
├── sobol_samples.npy
├── sobol_outputs.npz
├── metadata.json
└── timing.json
```

### Arrestatie-events

Elke rij in `arrests_XXXX.npy` is:

```python
(step, is_respawn, aggressiveness, is_home)
```

- `step`: 0–99, de measurement snapshot waarin de arrestatie plaatsvond;
- `is_respawn`: `False` voor een oorspronkelijke fan, `True` voor een vervangende fan;
- `aggressiveness`: agressiviteit van de gearresteerde fan;
- `is_home`: `True` voor HOME en `False` voor AWAY.

Meerdere arrestaties in dezelfde stap krijgen meerdere rijen met dezelfde `step`.
Een run zonder arrestaties krijgt een lege structured array.

### Entropy-overview

Elke `overview_XXXX.npy` bevat één rij:

```python
(run_id, valid, warmup_steps, warmup_entropy,
 start_measurement_entropy, end_measurement_entropy)
```

- `warmup_entropy`: fine spatial entropy direct na de warm-up;
- `start_measurement_entropy`: na 100 riot burn-in-stappen;
- `end_measurement_entropy`: na de 100e measurement-stap.

Na multiprocessing combineert het hoofdproces alle losse overview-bestanden tot `run_overview.npy`.

## Starten

N=32 basisdesign, 192 runs:

```bash
python run_sobol.py --workers 8 --base-n 32 --output-dir data
```

N=64, 384 runs:

```bash
python run_sobol.py --workers 64 --base-n 64 --output-dir data
```

N=128, 768 runs:

```bash
python run_sobol.py --workers 128 --base-n 128 --output-dir data
```

## Data openen

```python
import numpy as np

arrests = np.load("data/arrests/arrests_0042.npy", allow_pickle=False)
print(arrests.dtype.names)
print(arrests["step"])
print(arrests["aggressiveness"])

overview = np.load("data/run_overview.npy", allow_pickle=False)
print(overview[overview["run_id"] == 42])
```

## Snelle plots

```bash
python analyze_events.py --data-dir data --plots-dir plots_events
```

## Vereisten

- Python 3.10+
- NumPy
- Mesa
- SALib
- Matplotlib voor `analyze_events.py`
