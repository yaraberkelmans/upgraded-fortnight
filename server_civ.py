from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from SIMPLE_CIVIL.civil import (
    Citizen,
    CitizenState,
    Cop,
    CivilViolenceModel,
    CivilViolenceParams,
)


STATE_COLORS = {
    CitizenState.QUIET: "#4caf50",
    CitizenState.ACTIVE: "#d33f49",
}


def agent_portrayal(agent):
    portrayal = AgentPortrayalStyle(size=180)
    if isinstance(agent, Cop):
        portrayal.update(("color", "#111111"))
    elif isinstance(agent, Citizen):
        portrayal.update(("color", STATE_COLORS[agent.state]))
    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(8, 8)


def create_model(**kwargs):
    params = CivilViolenceParams(**kwargs)
    return CivilViolenceModel(params)


# Use an actual model instance for the renderer and for SolaraViz
initial_model = create_model()

renderer = SpaceRenderer(initial_model, backend="matplotlib").setup_agents(
    agent_portrayal
)
renderer.draw_agents()
renderer.post_process = post_process

chart_component = make_plot_component(
    {
        "Quiet": STATE_COLORS[CitizenState.QUIET],
        "Active": STATE_COLORS[CitizenState.ACTIVE],
    }
)

# Provide model_params in the Solara 'user inputs' format so sliders appear
model_params = {
    "N": {"type": "SliderInt", "value": 40, "min": 10, "max": 120, "step": 1, "label": "Grid size"},
    "citizen_density": {"type": "SliderFloat", "value": 0.70, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Citizen density"},
    "cop_density": {"type": "SliderFloat", "value": 0.074, "min": 0.0, "max": 0.5, "step": 0.001, "label": "Cop density"},
    "legitimacy": {"type": "SliderFloat", "value": 0.80, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Legitimacy"},
    "threshold": {"type": "SliderFloat", "value": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Threshold"},
    "citizen_vision": {"type": "SliderInt", "value": 7, "min": 1, "max": 20, "step": 1, "label": "Citizen vision"},
    "cop_vision": {"type": "SliderInt", "value": 7, "min": 1, "max": 20, "step": 1, "label": "Cop vision"},
    "steps": {"type": "SliderInt", "value": 200, "min": 1, "max": 2000, "step": 1, "label": "Steps"},
    "seed": {"type": "SliderInt", "value": 42, "min": 0, "max": 100000, "step": 1, "label": "Seed"},
}

page = SolaraViz(
    initial_model,
    renderer,
    components=[chart_component],
    model_params=model_params,
    name="Civil Violence Model",
)
page  # noqa: F401