# Three similarity-threshold time-to-arrest experiment

This is a small experiment separate from the full Sobol analysis. It runs
**three similarity-threshold setups ten times each**, for a default total of
**30 model runs**.

The agent density is fixed at `0.6`. All parameters remain equal between the
three setups except `similarity_threshold`.

Default thresholds:

- low: `0.25`
- medium: `0.45`
- high: `0.65`

You can change these values in `setups.json`.

## Cohort rule

After spatial warm-up and 100 riot burn-in steps, the current fan population is
frozen as the measurement cohort. Fans may already be respawns at this point.

During the following 100 measurement steps:

- only fans present at measurement start are analysed;
- each cohort fan can contribute at most one arrest time;
- respawns created during measurement are excluded;
- fans that are never arrested receive `arrest_step = -1`.

This prevents repeated respawns from inflating the arrest-time distribution.

## Install

```bash
pip install -r requirements.txt
```

## Run the 30 simulations

```bash
python run_three_setups.py --workers 8 --repeats 10 --seed 43
```

The same repetition seed is used for all three thresholds. This creates a
paired design and makes the threshold comparison less noisy.

## Create histograms and point plots

```bash
python plot_time_to_arrest.py --data-dir three_setup_data
```

Choose a different number of histogram bins with:

```bash
python plot_time_to_arrest.py --data-dir three_setup_data --bins 15
```

The following plots are saved in `three_setup_data/plots/`:

1. `time_to_arrest_histograms.png`
   - One histogram for each similarity threshold.
   - Arrest times from all ten runs are pooled within a setup.
   - The dashed line is the pooled mean arrest time.

2. `mean_time_to_arrest_points.png`
   - One point for the mean time to arrest in every individual run.
   - The diamond and error bar show the setup mean and one standard deviation.

3. `arrest_fraction_points.png`
   - One point for the fraction of the starting agents arrested in each run.
   - The diamond and error bar show the setup mean and one standard deviation.

4. `not_arrested_count_points.png`
   - One point for the number of agents present at measurement start that were not arrested.
   - The count is also stored per run in `run_summary.npy`, `run_summary.csv`, and `time_to_arrest_plot_data.npz`.

5. `mean_fighting_over_time.png`
   - Shows the mean number of fighting fans at every measurement step.
   - One line is shown per similarity threshold.
   - The shaded band represents one standard deviation across the ten runs.

The underlying values are also saved in:

```text
three_setup_data/time_to_arrest_plot_data.npz
```

## Output structure

```text
three_setup_data/
├── metadata.json
├── run_summary.npy
├── run_summary.csv
├── time_to_arrest_plot_data.npz
├── cohort/
│   ├── low_similarity_threshold/
│   ├── medium_similarity_threshold/
│   └── high_similarity_threshold/
├── fighting/
│   ├── low_similarity_threshold/
│   ├── medium_similarity_threshold/
│   └── high_similarity_threshold/
└── plots/
    ├── time_to_arrest_histograms.png
    ├── mean_time_to_arrest_points.png
    ├── arrest_fraction_points.png
    ├── not_arrested_count_points.png
    └── mean_fighting_over_time.png
```

## Histogram interpretation

The histogram title is:

**Time to Arrest by Similarity Threshold for Agents Present at Measurement Start (after warmup)**

Each bar shows the percentage of all agents present at measurement start that were arrested in that time interval. Each panel also reports the total percentage included in the histogram (arrested agents) and the percentage that was not arrested.


## Fighting time series

For every run, the number of fighting fans is saved after each of the 100 measurement steps in `three_setup_data/fighting/<setup>/`. The plotting script averages these series across repetitions and saves `mean_fighting_over_time.png`. The raw series, setup means, and standard deviations are also included in `time_to_arrest_plot_data.npz`.
