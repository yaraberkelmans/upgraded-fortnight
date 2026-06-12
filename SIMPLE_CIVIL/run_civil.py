import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import display

try:
    from SIMPLE_CIVIL.civil import CivilViolenceModel, CivilViolenceParams, CitizenState
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from civil import CivilViolenceModel, CivilViolenceParams, CitizenState


def plot_grid(model, title="Civil Violence grid"):
    quiet_x, quiet_y = [], []
    active_x, active_y = [], []
    cop_x, cop_y = [], []

    for citizen in model.citizens:
        x, y = citizen.pos

        if citizen.state == CitizenState.QUIET:
            quiet_x.append(x)
            quiet_y.append(y)
        elif citizen.state == CitizenState.ACTIVE:
            active_x.append(x)
            active_y.append(y)

    for cop in model.cops:
        x, y = cop.pos
        cop_x.append(x)
        cop_y.append(y)

    plt.figure(figsize=(7, 7))
    plt.scatter(quiet_x, quiet_y, s=20, label="Quiet citizens")
    plt.scatter(active_x, active_y, s=20, label="Active citizens")
    plt.scatter(cop_x, cop_y, s=30, label="Cops")
    plt.xlim(-1, model.params.N)
    plt.ylim(-1, model.params.N)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_results(results):
    ax = results[["Quiet", "Active"]].plot(figsize=(10, 5))
    ax.set_title("Epstein Civil Violence Model")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of citizens")
    plt.show()


def plot_risk_results(results):
    ax = results[["Average grievance", "Average arrest probability", "Average net risk"]].plot(figsize=(10, 5))
    ax.set_title("Average grievance, arrest probability and net risk")
    ax.set_xlabel("Step")
    ax.set_ylabel("Average value")
    plt.show()


def run_legitimacy_experiment(legitimacy, steps=200, seed=42):
    experiment_params = CivilViolenceParams(
        N=40,
        citizen_density=0.70,
        cop_density=0.074,
        legitimacy=legitimacy,
        threshold=0.10,
        citizen_vision=7,
        cop_vision=7,
        k=-math.log(0.1),
        steps=steps,
        seed=seed,
    )

    experiment_model = CivilViolenceModel(experiment_params)
    experiment_model.run_model()

    df = experiment_model.datacollector.get_model_vars_dataframe()
    df["Legitimacy"] = legitimacy
    return df


def main():
    # hier gaat de boel draaien, beetje rommelig maar werkt wel
    params = CivilViolenceParams(
        N=40,
        citizen_density=0.70,
        cop_density=0.004,
        legitimacy=0.80,
        threshold=0.10,
        citizen_vision=7,
        cop_vision=7,
        k=-math.log(0.1),
        steps=200,
        seed=42,
    )

    model = CivilViolenceModel(params)
    plot_grid(model, title="Starttoestand")

    model.run_model()

    results = model.datacollector.get_model_vars_dataframe()
    display(results.head())
    display(results.tail())

    # hier flikkert de plotzooi eruit, niet te netjes doen
    plot_results(results)
    plot_risk_results(results)
    plot_grid(model, title="Eindtoestand")

    df_high_legitimacy = run_legitimacy_experiment(legitimacy=0.85, steps=200, seed=42)
    df_low_legitimacy = run_legitimacy_experiment(legitimacy=0.65, steps=200, seed=42)

    plt.figure(figsize=(10, 5))
    plt.plot(df_high_legitimacy.index, df_high_legitimacy["Active"], label="Legitimacy = 0.85")
    plt.plot(df_low_legitimacy.index, df_low_legitimacy["Active"], label="Legitimacy = 0.65")
    plt.title("Effect van legitimacy op active citizens")
    plt.xlabel("Step")
    plt.ylabel("Number of active citizens")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()