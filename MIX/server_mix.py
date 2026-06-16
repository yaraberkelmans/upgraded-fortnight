"""Solara server for the mixed football-riot model.

Run with something like:
    solara run server_mix.py

Keep this file next to fan.py, cop.py and mixed_models.py, or adjust the imports
below to your package structure.
"""

from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from fan import Fan, FanGroup, FanState
from cop import Cop
from mixed_models import CrowdStrategy, MixedFootballRiotsModel, MixedFootballRiotsParams


FAN_COLORS = {
    (FanGroup.HOME, FanState.PASSIVE): "#2f80ed",      # home passive: blue
    (FanGroup.HOME, FanState.AGGRESSIVE): "#d33f49",   # home aggressive: red
    (FanGroup.AWAY, FanState.PASSIVE): "#f2c94c",      # away passive: yellow
    (FanGroup.AWAY, FanState.AGGRESSIVE): "#9b51e0",   # away aggressive: purple
}
COP_COLOR = "#111111"


STRATEGY_LABELS = {
    0: CrowdStrategy.NORMALIZATION.value,
    1: CrowdStrategy.MIXED.value,
    2: CrowdStrategy.SEPARATION.value,
}


def agent_portrayal(agent):
    """Visual style for every agent on the grid."""
    portrayal = AgentPortrayalStyle(size=160)

    if isinstance(agent, Cop):
        portrayal.update(("color", COP_COLOR))
    elif isinstance(agent, Fan):
        portrayal.update(("color", FAN_COLORS[(agent.group, agent.state)]))

    return portrayal


def post_process(ax):
    """Make the grid plot cleaner in Solara."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(8, 8)


def create_model(**kwargs):
    """Create a model from Solara UI values.

    A few UI controls are encoded as integers because that is robust with the
    same SliderInt style you already used in server_civ.py:
    - strategy_code: 0 normalization, 1 mixed, 2 separation
    - use_exponential_jump_int: 0 false, 1 true
    - use_schelling_movement_int: 0 false, 1 true
    - torus_int: 0 false, 1 true
    - max_jump_distance: 0 means no hard maximum.
    """
    strategy_code = int(kwargs.pop("strategy_code", 0))
    kwargs["strategy"] = STRATEGY_LABELS.get(strategy_code, CrowdStrategy.NORMALIZATION.value)

    kwargs["use_exponential_jump"] = bool(kwargs.pop("use_exponential_jump_int", 1))
    kwargs["use_schelling_movement"] = bool(kwargs.pop("use_schelling_movement_int", 1))
    kwargs["torus"] = bool(kwargs.pop("torus_int", 1))

    max_jump_distance = int(kwargs.get("max_jump_distance", 0))
    kwargs["max_jump_distance"] = None if max_jump_distance <= 0 else max_jump_distance

    # Keep the server compatible with older/newer versions of mixed_models.py.
    # Solara may pass every UI control, while the dataclass only accepts the
    # parameters that actually exist in your current MixedFootballRiotsParams.
    valid_keys = set(MixedFootballRiotsParams.__dataclass_fields__.keys())
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_keys}

    params = MixedFootballRiotsParams(**filtered_kwargs)
    return MixedFootballRiotsModel(params)


initial_model = create_model()

renderer = SpaceRenderer(initial_model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.draw_agents()
renderer.post_process = post_process

state_chart = make_plot_component(
    {
        "Passive home": FAN_COLORS[(FanGroup.HOME, FanState.PASSIVE)],
        "Aggressive home": FAN_COLORS[(FanGroup.HOME, FanState.AGGRESSIVE)],
        "Passive away": FAN_COLORS[(FanGroup.AWAY, FanState.PASSIVE)],
        "Aggressive away": FAN_COLORS[(FanGroup.AWAY, FanState.AGGRESSIVE)],
    }
)

def make_existing_plot_component(model, series_colors):
    """Create a Mesa plot only for columns that the model actually collects.

    This prevents pandas KeyError crashes when server_mix.py and mixed_models.py
    are temporarily out of sync. If a series is missing, update mixed_models.py
    or remove it from the requested list below.
    """
    df = model.datacollector.get_model_vars_dataframe()
    existing_series = {
        name: color
        for name, color in series_colors.items()
        if name in df.columns
    }
    return make_plot_component(existing_series)


mechanism_chart = make_existing_plot_component(
    initial_model,
    {
        "Fan-fan games": "#4f4f4f",
        "Average aggression score": "#d33f49",
        "Average local riot contagion": "#9b51e0",
        "Average arrest probability": "#111111",
    },
)

# Sliders/inputs shown in the Solara side panel.
model_params = {
    # Space and population.
    "N": {"type": "SliderInt", "value": 40, "min": 10, "max": 120, "step": 1, "label": "Grid size"},
    "fan_density": {"type": "SliderFloat", "value": 0.70, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Fan density"},
    "cop_density": {"type": "SliderFloat", "value": 0.074, "min": 0.0, "max": 0.5, "step": 0.001, "label": "Cop density"},
    "home_fraction": {"type": "SliderFloat", "value": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Home fraction"},
    "torus_int": {"type": "SliderInt", "value": 1, "min": 0, "max": 1, "step": 1, "label": "Torus? 0=no, 1=yes"},
    "seed": {"type": "SliderInt", "value": 42, "min": 0, "max": 100000, "step": 1, "label": "Seed"},
    "steps": {"type": "SliderInt", "value": 200, "min": 1, "max": 2000, "step": 1, "label": "Steps"},

    # Strategy.
    "strategy_code": {"type": "SliderInt", "value": 0, "min": 0, "max": 2, "step": 1, "label": "Strategy: 0=normalization, 1=mixed, 2=separation"},
    "separation_strength": {"type": "SliderFloat", "value": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Separation strength, only used when strategy=1"},

    # Vision and movement.
    "fan_vision": {"type": "SliderInt", "value": 7, "min": 1, "max": 20, "step": 1, "label": "Fan vision"},
    "cop_vision": {"type": "SliderInt", "value": 7, "min": 1, "max": 20, "step": 1, "label": "Cop vision"},
    "movement_radius": {"type": "SliderInt", "value": 7, "min": 1, "max": 20, "step": 1, "label": "Local movement radius"},
    "use_exponential_jump_int": {"type": "SliderInt", "value": 1, "min": 0, "max": 1, "step": 1, "label": "Exponential jump? 0=no, 1=yes"},
    "jump_decay": {"type": "SliderFloat", "value": 0.35, "min": 0.01, "max": 2.0, "step": 0.01, "label": "Jump decay"},
    "max_jump_distance": {"type": "SliderInt", "value": 0, "min": 0, "max": 120, "step": 1, "label": "Max jump distance, 0=no max"},

    # Schelling-like movement.
    "use_schelling_movement_int": {"type": "SliderInt", "value": 1, "min": 0, "max": 1, "step": 1, "label": "Schelling movement? 0=no, 1=yes"},
    "similarity_threshold": {"type": "SliderFloat", "value": 0.30, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Similarity threshold"},

    # Fan-fan game / aggression score.
    "aggression_threshold": {"type": "SliderFloat", "value": 0.10, "min": -2.0, "max": 2.0, "step": 0.01, "label": "Aggression threshold"},
    "hostility_weight": {"type": "SliderFloat", "value": 0.50, "min": 0.0, "max": 3.0, "step": 0.01, "label": "Personal hostility weight"},
    "home_away_aggression": {"type": "SliderFloat", "value": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Home-away aggression"},
    "relationship_weight": {"type": "SliderFloat", "value": 1.00, "min": 0.0, "max": 3.0, "step": 0.01, "label": "Relationship weight"},
    "rival_weight": {"type": "SliderFloat", "value": 1.00, "min": 0.0, "max": 3.0, "step": 0.01, "label": "Rival presence weight"},
    "riot_contagion_weight": {"type": "SliderFloat", "value": 0.70, "min": 0.0, "max": 3.0, "step": 0.01, "label": "Riot contagion weight"},
    "own_group_buffer": {"type": "SliderFloat", "value": 0.30, "min": -2.0, "max": 2.0, "step": 0.01, "label": "Own-group buffer, negative = group confidence"},
    "police_deterrence": {"type": "SliderFloat", "value": 0.80, "min": 0.0, "max": 3.0, "step": 0.01, "label": "Police deterrence"},
    "risk_weight": {"type": "SliderFloat", "value": 1.00, "min": 0.0, "max": 3.0, "step": 0.01, "label": "Arrest-risk weight"},
    "k": {"type": "SliderFloat", "value": 2.302585093, "min": 0.01, "max": 10.0, "step": 0.01, "label": "Arrest-risk k"},
}

page = SolaraViz(
    initial_model,
    renderer,
    components=[state_chart, mechanism_chart],
    model_params=model_params,
    name="Mixed Football Riots Model",
)

page  # noqa: F401
