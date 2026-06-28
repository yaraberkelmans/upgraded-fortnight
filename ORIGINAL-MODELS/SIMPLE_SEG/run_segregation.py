import sys
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from SIMPLE_CIVIL.segregation import SegregationModel, SegregationParams, Group
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from segregation import SegregationModel, SegregationParams, Group


try:
    from IPython.display import display
except ImportError:
    display = print


def plot_grid(model, title="Schelling segregation grid"):
    red_x, red_y = [], []
    green_x, green_y = [], []

    for household in model.households:
        x, y = household.pos
        if household.group == Group.RED:
            red_x.append(x)
            red_y.append(y)
        elif household.group == Group.GREEN:
            green_x.append(x)
            green_y.append(y)

    plt.figure(figsize=(7, 7))
    plt.scatter(red_x, red_y, s=20, label="Red households")
    plt.scatter(green_x, green_y, s=20, label="Green households")
    plt.xlim(-1, model.params.N)
    plt.ylim(-1, model.params.N)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_happiness_results(results):
    ax = results[["Happy", "Unhappy"]].plot(figsize=(10, 5))
    ax.set_title("Schelling Segregation Model: happy vs unhappy households")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of households")
    plt.show()


def plot_similarity_results(results):
    ax = results[["Average similarity", "Moves"]].plot(figsize=(10, 5), secondary_y="Moves")
    ax.set_title("Average local similarity and moves per step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Average similarity")
    ax.right_ax.set_ylabel("Moves")
    plt.show()


def run_threshold_experiment(similarity_threshold, steps=100, seed=42):
    params = SegregationParams(
        N=40,
        agent_density=0.80,
        red_fraction=0.50,
        similarity_threshold=similarity_threshold,
        steps=steps,
        seed=seed,
        torus=True,
        count_empty_as_different=True,
    )

    model = SegregationModel(params)
    model.run_model()

    df = model.datacollector.get_model_vars_dataframe()
    df["Similarity threshold"] = similarity_threshold
    return df


def main():
    params = SegregationParams(
        N=40,
        agent_density=0.80,
        red_fraction=0.50,
        similarity_threshold=0.60,
        steps=100,
        seed=42,
        torus=True,
        count_empty_as_different=True,
    )

    model = SegregationModel(params)
    plot_grid(model, title="Starttoestand")

    model.run_model()

    results = model.datacollector.get_model_vars_dataframe()
    display(results.head())
    display(results.tail())

    plot_happiness_results(results)
    plot_similarity_results(results)
    plot_grid(model, title="Eindtoestand")

    low_threshold = run_threshold_experiment(similarity_threshold=0.25, steps=100, seed=42)
    high_threshold = run_threshold_experiment(similarity_threshold=0.50, steps=100, seed=42)

    plt.figure(figsize=(10, 5))
    plt.plot(
        low_threshold.index,
        low_threshold["Average similarity"],
        label="Threshold = 0.25",
    )
    plt.plot(
        high_threshold.index,
        high_threshold["Average similarity"],
        label="Threshold = 0.50",
    )
    plt.title("Effect van similarity threshold op segregatie")
    plt.xlabel("Step")
    plt.ylabel("Average similarity")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
