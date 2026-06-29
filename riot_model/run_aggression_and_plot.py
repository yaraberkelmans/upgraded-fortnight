
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from riot_model import RiotModel, RiotParams, SegregationParams

@dataclass
class Config:
    # Welke parameter wil je onderzoeken?
    sweep_parameter: str = "home_fraction"

    # Waarden van de sweep.
    sweep_start: float = 0.0
    sweep_stop: float = 1.00
    sweep_step: float = 0.2

    repetitions: int = 3
    model_steps: int = 100
    base_seed: int = 42

    # Vaste segregatieparameters.
    N: int = 40
    agent_density: float = 0.80
    home_fraction: float = 0.50
    similarity_threshold: float = 0.30
    movement_decay: float = 1.0
    torus: bool = True
    count_empty_as_different: bool = True

    # Vaste riotparameters.
    police_density: float = 0.05
    perception_k: float = 0.693
    fan_vision: int = 2
    fight_threshold: float = 0.0
    police_vision: int = 5
    hawk_dove_C: float = 4.0
    aggressiveness_mean: float | None = None
    aggressiveness_concentration: float = 12.0

    output_dir: str = "data/"


CFG = Config()


SEGREGATION_FIELDS = {
    field.name: field.type
    for field in fields(SegregationParams)
}

RIOT_FIELDS = {
    field.name: field.type
    for field in fields(RiotParams)
}

INTEGER_PARAMETERS = {
    "N",
    "steps",
    "seed",
    "fan_vision",
    "police_vision",
}


def sweep_values():
    if CFG.sweep_step == 0:
        raise ValueError("sweep_step mag niet 0 zijn.")

    values = []
    value = CFG.sweep_start
    tolerance = abs(CFG.sweep_step) * 1e-9 + 1e-12

    if CFG.sweep_step > 0:
        while value <= CFG.sweep_stop + tolerance:
            values.append(round(value, 10))
            value += CFG.sweep_step
    else:
        while value >= CFG.sweep_stop - tolerance:
            values.append(round(value, 10))
            value += CFG.sweep_step

    return values


def cast_sweep_value(value):
    if CFG.sweep_parameter in INTEGER_PARAMETERS:
        return int(round(value))
    return value


def build_params(sweep_value, seed):
    segregation_kwargs = {
        "N": CFG.N,
        "agent_density": CFG.agent_density,
        "home_fraction": CFG.home_fraction,
        "similarity_threshold": CFG.similarity_threshold,
        "movement_decay": CFG.movement_decay,
        "steps": CFG.model_steps,
        "seed": seed,
        "torus": CFG.torus,
        "count_empty_as_different": CFG.count_empty_as_different,
    }

    riot_kwargs = {
        "police_density": CFG.police_density,
        "perception_k": CFG.perception_k,
        "fan_vision": CFG.fan_vision,
        "fight_threshold": CFG.fight_threshold,
        "police_vision": CFG.police_vision,
        "hawk_dove_C": CFG.hawk_dove_C,
        "aggressiveness_mean": CFG.aggressiveness_mean,
        "aggressiveness_concentration": CFG.aggressiveness_concentration,
    }

    value = cast_sweep_value(sweep_value)

    if CFG.sweep_parameter in segregation_kwargs:
        segregation_kwargs[CFG.sweep_parameter] = value
    elif CFG.sweep_parameter in riot_kwargs:
        riot_kwargs[CFG.sweep_parameter] = value
    else:
        valid = sorted(
            set(segregation_kwargs) | set(riot_kwargs)
        )
        raise ValueError(
            f"Onbekende sweep_parameter: {CFG.sweep_parameter}\n"
            f"Kies uit: {', '.join(valid)}"
        )

    return (
        SegregationParams(**segregation_kwargs),
        RiotParams(**riot_kwargs),
    )


def run_experiment():
    rows = []

    for repetition in range(1, CFG.repetitions + 1):
        seed = CFG.base_seed + repetition - 1

        for raw_value in sweep_values():
            segregation, riot = build_params(raw_value, seed)

            model = RiotModel(
                segregation_params=segregation,
                riot_params=riot,
            )

            initial_fans = len(model.fans)
            total_arrests = 0

            for _ in range(CFG.model_steps):
                model.step()
                total_arrests += model.arrests_this_step

            remaining_fans = len(model.fans)

            rows.append({
                "repetition": repetition,
                "seed": seed,
                "parameter": CFG.sweep_parameter,
                "parameter_value": cast_sweep_value(raw_value),
                "total_arrests": total_arrests,
                "initial_fans": initial_fans,
                "remaining_fans": remaining_fans,
                "arrest_fraction": (
                    total_arrests / initial_fans
                    if initial_fans else 0.0
                ),
            })

            print(
                f"run={repetition} "
                f"seed={seed} "
                f"{CFG.sweep_parameter}={cast_sweep_value(raw_value)} "
                f"opgepakt={total_arrests}"
            )

    return pd.DataFrame(rows)


def make_plots(df, output_dir):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("parameter_value", as_index=False)
        .agg(
            runs=("total_arrests", "count"),
            mean_total_arrests=("total_arrests", "mean"),
            std_total_arrests=("total_arrests", "std"),
            min_total_arrests=("total_arrests", "min"),
            max_total_arrests=("total_arrests", "max"),
            mean_arrest_fraction=("arrest_fraction", "mean"),
            std_arrest_fraction=("arrest_fraction", "std"),
            mean_remaining_fans=("remaining_fans", "mean"),
        )
        .fillna(0.0)
        .sort_values("parameter_value")
    )

    summary.to_csv(
        output_dir / f"{CFG.sweep_parameter}_summary.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    for repetition, group in df.groupby("repetition"):
        group = group.sort_values("parameter_value")
        plt.plot(
            group["parameter_value"],
            group["total_arrests"],
            marker="o",
            label=f"Run {repetition}",
        )
    plt.xlabel(CFG.sweep_parameter)
    plt.ylabel("Totaal opgepakte fans")
    plt.title(f"Opgepakte fans per run: {CFG.sweep_parameter}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plots_dir / "arrests_per_run.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["parameter_value"],
        summary["mean_total_arrests"],
        yerr=summary["std_total_arrests"],
        marker="o",
        capsize=5,
    )
    plt.xlabel(CFG.sweep_parameter)
    plt.ylabel("Gemiddeld aantal opgepakte fans")
    plt.title(f"Gemiddelde arrestaties: {CFG.sweep_parameter}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        plots_dir / "mean_arrests.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["parameter_value"],
        summary["mean_arrest_fraction"],
        yerr=summary["std_arrest_fraction"],
        marker="o",
        capsize=5,
    )
    plt.xlabel(CFG.sweep_parameter)
    plt.ylabel("Gemiddelde arrestfractie")
    plt.title(f"Arrestfractie: {CFG.sweep_parameter}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        plots_dir / "arrest_fraction.png",
        dpi=150,
    )
    plt.close()

    return summary


def main():
    output_dir = (
        Path(CFG.output_dir)
        / CFG.sweep_parameter
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment()

    csv_path = (
        output_dir
        / f"{CFG.sweep_parameter}_experiment.csv"
    )
    results.to_csv(csv_path, index=False)

    summary = make_plots(results, output_dir)

    print("\nKlaar")
    print(f"CSV: {csv_path}")
    print(f"Plots: {output_dir / 'plots'}")
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
