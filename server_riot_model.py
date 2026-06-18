"""Solara server for the riot_model package.

Run with something like:
    solara run server_riot_model.py
"""

import matplotlib.pyplot as plt
import solara
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from riot_model.riot_model import Fan, FanGroup, HawkDoveStrategy, Police, RiotModel, SegregationParams
STATE_COLORS = {
    FanGroup.HOME: "#2f80ed",
    FanGroup.AWAY: "#f2c94c",
    "POLICE": "#111111",
}


def agent_portrayal(agent):
    size = 120 + min(agent.last_move_distance, 6) * 25
    portrayal = AgentPortrayalStyle(size=size)
    if isinstance(agent, Fan):
        if agent.fighting:
            portrayal.update(("color", "#eb5757"))
        else:
            portrayal.update(("color", STATE_COLORS[agent.group]))
    elif isinstance(agent, Police):
        portrayal.update(("color", STATE_COLORS["POLICE"]))
    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(8, 8)


initial_model = RiotModel()

renderer = SpaceRenderer(initial_model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.draw_agents()
renderer.post_process = post_process

chart_component = make_plot_component(
    {
        "Happy": "#4caf50",
        "Unhappy": "#d33f49",
        "Fighting fans": "#eb5757",
    }
)

movement_chart = make_plot_component(
    {
        "Average last move distance": "#111111",
        "Average last move distance (moved fans)": "#9b51e0",
    }
)

arrest_chart = make_plot_component(
    {
        "Arrests this step": "#27ae60",
    }
)

aggressiveness_chart = make_plot_component(
    {
        "Average aggressiveness": "#f2994a",
    }
)

perception_chart = make_plot_component(
    {
        "Average perceived win probability": "#2f80ed",
        "Average perceived arrest probability": "#27ae60",
    }
)


@solara.component
def FightDistributions(model):
    current_model = model.value if hasattr(model, "value") else model
    fight_want = [fan.fight_want for fan in current_model.fans]
    fight_margin = [fan.fight_want - fan.perceived_arrest_probability for fan in current_model.fans]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(fight_want, bins=30, color="#f2994a", edgecolor="white", alpha=0.85)
    ax1.set_title("fight_want distribution")
    ax1.set_xlabel("fight_want")
    ax1.set_ylabel("fans")

    ax2.hist(fight_margin, bins=30, color="#eb5757", edgecolor="white", alpha=0.85)
    ax2.set_title("fight_want − P(arrest) distribution")
    ax2.set_xlabel("fight_want − P(arrest)")
    ax2.set_ylabel("fans")
    ax2.axvline(current_model.riot_params.fight_threshold, color="#111", linestyle="--", linewidth=1, label="threshold")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)

model_params = {
    "N": {"type": "SliderInt", "value": 40, "min": 10, "max": 120, "step": 1, "label": "Grid size"},
    "agent_density": {"type": "SliderFloat", "value": 0.80, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Agent density"},
    "police_density": {"type": "SliderFloat", "value": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Police density"},
    "home_fraction": {"type": "SliderFloat", "value": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Home fraction"},
    "similarity_threshold": {"type": "SliderFloat", "value": 0.30, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Similarity threshold"},
    "fan_vision": {"type": "SliderInt", "value": 2, "min": 1, "max": 20, "step": 1, "label": "Fan vision"},
    "perception_k": {"type": "SliderFloat", "value": 0.693, "min": 0.0, "max": 5.0, "step": 0.01, "label": "Perception k"},
    "fight_threshold": {"type": "SliderFloat", "value": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "label": "Fight threshold"},
    "police_vision": {"type": "SliderInt", "value": 5, "min": 1, "max": 20, "step": 1, "label": "Police vision"},
    "hawk_dove_strategy": {
        "type": "Select",
        "value": "aggressiveness",
        "values": [s.value for s in HawkDoveStrategy],
        "label": "Hawk-Dove strategy",
    },
    "hawk_dove_C": {"type": "SliderFloat", "value": 4.0, "min": 0.1, "max": 10.0, "step": 0.1, "label": "Hawk-Dove C (injury cost)"},
    "movement_decay": {"type": "SliderFloat", "value": 1.00, "min": 0.01, "max": 5.0, "step": 0.01, "label": "Movement decay"},
    "steps": {"type": "SliderInt", "value": 100, "min": 1, "max": 2000, "step": 1, "label": "Steps"},
    "seed": {"type": "SliderInt", "value": 42, "min": 0, "max": 100000, "step": 1, "label": "Seed"},
    "torus": {"type": "Checkbox", "value": True, "label": "Torus"},
    "count_empty_as_different": {"type": "Checkbox", "value": True, "label": "Count empty as different"},
}

page = SolaraViz(
    initial_model,
    renderer,
    components=[chart_component, arrest_chart, movement_chart, aggressiveness_chart, perception_chart, FightDistributions],
    model_params=model_params,
    name="Mix Start Fan Model",
)

page  # noqa: F401
