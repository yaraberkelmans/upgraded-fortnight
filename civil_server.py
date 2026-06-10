from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from SIMPLE_CIVIL.civil import Citizen, CitizenState, Cop, CivilViolenceModel


STATE_COLORS = {
    CitizenState.QUIET: "#4caf50",
    CitizenState.ACTIVE: "#d33f49",
    CitizenState.ARRESTED: "#999999",
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


model = CivilViolenceModel()

renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.draw_agents()
renderer.post_process = post_process

chart_component = make_plot_component(
    {
        "Quiet": STATE_COLORS[CitizenState.QUIET],
        "Active": STATE_COLORS[CitizenState.ACTIVE],
        "Arrested": STATE_COLORS[CitizenState.ARRESTED],
    }
)

page = SolaraViz(
    model,
    renderer,
    components=[chart_component],
    model_params={},
    name="Civil Violence Model",
)
page  # noqa: F401
