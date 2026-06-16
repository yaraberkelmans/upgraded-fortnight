from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from SIMPLE_SEG.segregation import (
    Household,
    Group,
    SegregationModel,
    SegregationParams,
)

STATE_COLORS = {
    Group.RED: "#d33f49",
    Group.GREEN: "#4caf50",
}


def agent_portrayal(agent):
    portrayal = AgentPortrayalStyle(size=180)
    if isinstance(agent, Household):
        portrayal.update(("color", STATE_COLORS[agent.group]))
    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(8, 8)


# create an initial model instance
initial_model = SegregationModel()

renderer = SpaceRenderer(initial_model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.draw_agents()
renderer.post_process = post_process

chart_component = make_plot_component(
    {
        "Happy": "#4caf50",
        "Unhappy": "#d33f49",
    }
)

model_params = {
    "N": {"type": "SliderInt", "value": 40, "min": 10, "max": 120, "step": 1, "label": "Grid size"},
    "agent_density": {"type": "SliderFloat", "value": 0.80, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Agent density"},
    "red_fraction": {"type": "SliderFloat", "value": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Red fraction"},
    "similarity_threshold": {"type": "SliderFloat", "value": 0.30, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Similarity threshold"},
    "steps": {"type": "SliderInt", "value": 100, "min": 1, "max": 2000, "step": 1, "label": "Steps"},
    "seed": {"type": "SliderInt", "value": 42, "min": 0, "max": 100000, "step": 1, "label": "Seed"},
    "torus": {"type": "Checkbox", "value": True, "label": "Torus"},
    "count_empty_as_different": {"type": "Checkbox", "value": True, "label": "Count empty as different"},
}

page = SolaraViz(
    initial_model,
    renderer,
    components=[chart_component],
    model_params=model_params,
    name="Segregation Model",
)
page  # noqa: F401