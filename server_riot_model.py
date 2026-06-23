"""Solara server for the riot_model package.

Run with something like:
    solara run server_riot_model.py
"""

import solara
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from riot_model.riot_model import Fan, FanGroup, HawkDoveStrategy, Police, RiotModel
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

entropy_chart = make_plot_component(
    {
        "Spatial entropy": "#9b51e0",
        "Spatial entropy (fine)": "#2f80ed",
    }
)

entropy_cv_chart = make_plot_component(
    {
        "Entropy CV": "#9b51e0",
        "Entropy CV (fine)": "#2f80ed",
    }
)

perception_chart = make_plot_component(
    {
        "Average perceived win probability": "#2f80ed",
        "Average perceived arrest probability": "#27ae60",
    }
)


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
        "value": "logit_qre",
        "values": [s.value for s in HawkDoveStrategy],
        "label": "Hawk-Dove strategy",
    },
    "hawk_dove_C": {"type": "SliderFloat", "value": 4.0, "min": 0.1, "max": 10.0, "step": 0.1, "label": "Hawk-Dove C (injury cost)"}, "logit_beta": {"type": "SliderFloat", "value": 5.0, "min": 0.1, "max": 20.0, "step": 0.1, "label": "Logit beta (rationality)"},
    "movement_decay": {"type": "SliderFloat", "value": 1.00, "min": 0.01, "max": 5.0, "step": 0.01, "label": "Movement decay"},
    "steps": {"type": "SliderInt", "value": 100, "min": 1, "max": 2000, "step": 1, "label": "Steps"},
    "seed": {"type": "SliderInt", "value": 42, "min": 0, "max": 100000, "step": 1, "label": "Seed"},
    "torus": {"type": "Checkbox", "value": True, "label": "Torus"},
    "count_empty_as_different": {"type": "Checkbox", "value": True, "label": "Count empty as different"},
    "random_move_chance": {"type": "SliderFloat", "value": 0.005, "min": 0.001, "max": 0.15, "step": 0.001, "label": "Random move chance"},
    "warmup_cv_threshold": {"type": "SliderFloat", "value": 0.01, "min": 0.001, "max": 0.1, "step": 0.001, "label": "Warmup CV threshold"},
    "fighting_enabled": {"type": "Checkbox", "value": True, "label": "Fighting enabled"},
}

page = SolaraViz(
    initial_model,
    renderer,
    components=[chart_component, arrest_chart, movement_chart, aggressiveness_chart, perception_chart, entropy_chart, entropy_cv_chart],
    model_params=model_params,
    name="Mix Start Fan Model",
)

page  # noqa: F401
