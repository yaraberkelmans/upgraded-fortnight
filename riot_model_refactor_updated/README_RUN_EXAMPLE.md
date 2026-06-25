# Run example

Run from this directory:

```bash
python run_example.py
```

The script performs:

1. spatial warm-up until the existing fine-entropy CV rule exits;
2. 100 fixed riot burn-in steps without recording rows;
3. 100 measurement steps stored in `system_state_100_steps.npy`.

Outputs are written to `run_example_output/`:

- `system_state_100_steps.npy`: structured NumPy array with exactly 100 rows;
- `timings.json`: execution time for initialization and all phases;
- `parameters.json`: parameters used for the run;
- `summary.json`: aggregate values for later Sobol analysis;
- `plots/`: eight PNG plots covering the recorded system metrics;
- `manifest.json`: paths to all generated files.

Load the array with:

```python
import numpy as np

state = np.load(
    "run_example_output/system_state_100_steps.npy",
    allow_pickle=False,
)
print(state.dtype.names)
print(state["fighting"])
```
