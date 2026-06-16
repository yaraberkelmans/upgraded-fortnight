"""Mixed Civil Violence + Schelling model for football riots.

The model separates Fan and Cop classes into fan.py and cop.py. It can be used
for Solara later because the model accepts either a params dataclass or keyword
arguments.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid

from cop import Cop
from fan import Fan, FanGroup, FanState


class CrowdStrategy(Enum):
    """Policy regime for the spatial environment."""

    NORMALIZATION = "normalization"
    SEPARATION = "separation"
    MIXED = "mixed"


@dataclass
class MixedFootballRiotsParams:
    # Space and population.
    N: int = 40
    fan_density: float = 0.70
    cop_density: float = 0.074
    home_fraction: float = 0.50
    torus: bool = True
    seed: int = 42
    steps: int = 200

    # Spatial strategy. 0 = normalization/free mixing, 1 = strong separation.
    strategy: str = CrowdStrategy.NORMALIZATION.value
    separation_strength: float = 0.0

    # Fan perception and movement.
    fan_vision: int = 7
    cop_vision: int = 7
    movement_radius: int = 7
    use_exponential_jump: bool = True
    jump_decay: float = 0.35
    max_jump_distance: int | None = None

    # Schelling-like clustering movement.
    use_schelling_movement: bool = True
    similarity_threshold: float = 0.30
    count_empty_as_different: bool = True

    # Civil Violence / fan-fan game parameters.
    aggression_threshold: float = 0.10
    hostility_weight: float = 0.50
    home_away_aggression: float = 0.50
    relationship_weight: float = 1.00
    rival_weight: float = 1.00
    riot_contagion_weight: float = 0.70
    own_group_buffer: float = 0.30
    police_deterrence: float = 0.80
    risk_weight: float = 1.00
    k: float = -math.log(0.1)

    def __post_init__(self):
        if isinstance(self.strategy, CrowdStrategy):
            self.strategy = self.strategy.value

        if self.strategy == CrowdStrategy.NORMALIZATION.value:
            self.separation_strength = 0.0
        elif self.strategy == CrowdStrategy.SEPARATION.value:
            self.separation_strength = 1.0
        else:
            self.separation_strength = min(1.0, max(0.0, self.separation_strength))


class MixedFootballRiotsModel(Model):
    """Football-riot model with explicit fan and police agent classes."""

    def __init__(self, params: MixedFootballRiotsParams | None = None, **kwargs):
        super().__init__()

        if params is None:
            valid_keys = set(MixedFootballRiotsParams.__dataclass_fields__.keys())
            filtered = {key: value for key, value in kwargs.items() if key in valid_keys}
            self.params = MixedFootballRiotsParams(**filtered)
        else:
            self.params = params

        self.random = random.Random(self.params.seed)
        self.grid = SingleGrid(
            width=self.params.N,
            height=self.params.N,
            torus=self.params.torus,
        )

        self.fans: list[Fan] = []
        self.cops: list[Cop] = []
        self.arrested_fans: list[Fan] = []
        self.moves_this_step = 0
        self.arrests_this_step = 0
        self.fan_games_this_step = 0

        self.create_agents()
        self.update_all_fan_states()

        self.datacollector = DataCollector(
            model_reporters={
                "Passive home": lambda m: m.count_fans(FanGroup.HOME, FanState.PASSIVE),
                "Aggressive home": lambda m: m.count_fans(FanGroup.HOME, FanState.AGGRESSIVE),
                "Passive away": lambda m: m.count_fans(FanGroup.AWAY, FanState.PASSIVE),
                "Aggressive away": lambda m: m.count_fans(FanGroup.AWAY, FanState.AGGRESSIVE),
                "Aggressive total": lambda m: m.count_state(FanState.AGGRESSIVE),
                "Cops": lambda m: len(m.cops),
                "Arrests": lambda m: m.arrests_this_step,
                "Fan-fan games": lambda m: m.fan_games_this_step,
                "Home-away aggression": lambda m: m.params.home_away_aggression,
                "Average relationship aggression": lambda m: m.average_relationship_aggression(),
                "Average local riot contagion": lambda m: m.average_local_riot_contagion(),
                "Average aggression score": lambda m: m.average_aggression_score(),
                "Average arrest probability": lambda m: m.average_arrest_probability(),
                "Average similarity": lambda m: m.average_similarity(),
                "Moves": lambda m: m.moves_this_step,
            }
        )
        self.datacollector.collect(self)

    def all_positions(self):
        return [(x, y) for x in range(self.params.N) for y in range(self.params.N)]

    def split_positions_by_side(self, positions):
        middle = self.params.N / 2
        home_side = [pos for pos in positions if pos[0] < middle]
        away_side = [pos for pos in positions if pos[0] >= middle]
        return home_side, away_side

    def create_agents(self) -> None:
        total_cells = self.params.N * self.params.N
        number_of_fans = int(total_cells * self.params.fan_density)
        number_of_cops = int(total_cells * self.params.cop_density)
        number_of_home = int(number_of_fans * self.params.home_fraction)
        number_of_away = number_of_fans - number_of_home

        positions = self.all_positions()
        self.random.shuffle(positions)

        if self.params.separation_strength >= 0.5:
            home_positions, away_positions = self.split_positions_by_side(positions)
            self.random.shuffle(home_positions)
            self.random.shuffle(away_positions)
            fan_positions = (
                [(pos, FanGroup.HOME) for pos in home_positions[:number_of_home]]
                + [(pos, FanGroup.AWAY) for pos in away_positions[:number_of_away]]
            )
            used_positions = {pos for pos, _ in fan_positions}
            remaining_positions = [pos for pos in positions if pos not in used_positions]
        else:
            groups = [FanGroup.HOME] * number_of_home + [FanGroup.AWAY] * number_of_away
            self.random.shuffle(groups)
            fan_positions = list(zip(positions[:number_of_fans], groups))
            remaining_positions = positions[number_of_fans:]

        for pos, group in fan_positions:
            fan = Fan(self, group)
            self.fans.append(fan)
            self.grid.place_agent(fan, pos)

        # In separation, police start closer to the border; in normalization,
        # remaining positions are simply shuffled.
        if self.params.separation_strength >= 0.5:
            remaining_positions.sort(key=lambda pos: abs(pos[0] - ((self.params.N - 1) / 2)))
            border_pool = remaining_positions[: max(number_of_cops * 3, number_of_cops)]
            self.random.shuffle(border_pool)
            cop_positions = border_pool[:number_of_cops]
        else:
            self.random.shuffle(remaining_positions)
            cop_positions = remaining_positions[:number_of_cops]

        for pos in cop_positions:
            cop = Cop(self)
            self.cops.append(cop)
            self.grid.place_agent(cop, pos)

    def update_all_fan_states(self) -> None:
        for fan in self.fans[:]:
            fan.decide_state()
            fan.decide_happiness()

    def relationship_aggression(self, source_group: FanGroup, target_group: FanGroup) -> float:
        """Directed supporter-relation value used in the fan aggression score.

        With two groups, the rival mapping is symmetric:
        HOME -> AWAY and AWAY -> HOME. Same-group relations have value 0.
        For now this is a single model-level variable, so the HOME-AWAY and
        AWAY-HOME relation have the same value. Later this can become a matrix
        if one side should be more hostile than the other.
        """
        if source_group == target_group:
            return 0.0
        if {source_group, target_group} == {FanGroup.HOME, FanGroup.AWAY}:
            return self.params.home_away_aggression
        return 0.0

    def count_fan_games(self) -> int:
        """Count local rival encounters once per pair.

        This approximates how many fan-fan games are available in the step.
        """
        seen_pairs = set()
        games = 0
        for fan in self.fans:
            neighbours = self.grid.get_neighbors(
                fan.pos,
                moore=True,
                include_center=False,
                radius=1,
            )
            for other in neighbours:
                if isinstance(other, Fan) and other.group != fan.group:
                    pair = tuple(sorted((id(fan), id(other))))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        games += 1
        return games

    def step(self) -> None:
        self.moves_this_step = 0
        self.arrests_this_step = 0
        self.fan_games_this_step = self.count_fan_games()

        agents = self.fans[:] + self.cops[:]
        self.random.shuffle(agents)

        for agent in agents:
            # A fan may have been arrested earlier in the same step.
            if isinstance(agent, Fan) and agent not in self.fans:
                continue
            agent.step()

        self.update_all_fan_states()
        self.fan_games_this_step = self.count_fan_games()
        self.datacollector.collect(self)

    def run_model(self, steps=None) -> None:
        if steps is None:
            steps = self.params.steps
        for _ in range(steps):
            self.step()

    def count_fans(self, group: FanGroup, state: FanState | None = None) -> int:
        if state is None:
            return sum(fan.group == group for fan in self.fans)
        return sum(fan.group == group and fan.state == state for fan in self.fans)

    def count_state(self, state: FanState) -> int:
        return sum(fan.state == state for fan in self.fans)

    def average_relationship_aggression(self) -> float:
        if not self.fans:
            return 0.0
        return sum(fan.relationship_aggression for fan in self.fans) / len(self.fans)

    def average_local_riot_contagion(self) -> float:
        if not self.fans:
            return 0.0
        return sum(fan.local_aggressive_fraction for fan in self.fans) / len(self.fans)

    def average_aggression_score(self) -> float:
        if not self.fans:
            return 0.0
        return sum(fan.aggression_score for fan in self.fans) / len(self.fans)

    def average_arrest_probability(self) -> float:
        if not self.fans:
            return 0.0
        return sum(fan.arrest_probability for fan in self.fans) / len(self.fans)

    def average_similarity(self) -> float:
        if not self.fans:
            return 0.0
        return sum(fan.same_fraction for fan in self.fans) / len(self.fans)
