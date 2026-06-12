from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from SIMPLE_SEG.segregation import Household, Group, SegregationModel

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


model = SegregationModel()

renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.draw_agents()
renderer.post_process = post_process

chart_component = make_plot_component(
    {
        "Happy": "#4caf50",
        "Unhappy": "#d33f49",
    }
)

page = SolaraViz(
    model,
    renderer,
    components=[chart_component],
    model_params={},
    name="Segregation Model",
)
page  # noqa: F401