"""Main Mesa riot model. Agent implementations live in fan.py and police.py."""

import random
import math
from dataclasses import dataclass

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid

try:    
    from riot_model.fan import Fan, FanGroup, HawkDoveStrategy
    from riot_model.police import Police
except ImportError:   
    from fan import Fan, FanGroup, HawkDoveStrategy
    from police import Police

@dataclass
class SegregationParams:
    N: int = 40
    agent_density: float = 0.80
    home_fraction: float = 0.50
    similarity_threshold: float = 0.30
    movement_decay: float = 1.0
    steps: int = 100
    seed: int = 42
    torus: bool = True
    count_empty_as_different: bool = True
    zone_size: int = 10
    zone_size_fine: int = 4
    warmup_cv_threshold: float = 0.01
    warmup_window: int = 10
    random_move_chance: float = 0.005


@dataclass
class RiotParams:
    police_density: float = 0.05
    perception_k: float = 0.693
    fan_vision: int = 2
    fight_threshold: float = 0.0
    police_vision: int = 5
    logit_beta: float = 5.0
    hawk_dove_strategy: HawkDoveStrategy = HawkDoveStrategy.LOGIT
    hawk_dove_C: float = 4.0
    aggressiveness_mean: float | None = None
    aggressiveness_concentration: float = 12.0
    fighting_enabled: bool = True


    def __post_init__(self):
        if isinstance(self.hawk_dove_strategy, str):
            self.hawk_dove_strategy = HawkDoveStrategy(self.hawk_dove_strategy)
        if self.aggressiveness_mean is not None and not 0.0 <= self.aggressiveness_mean <= 1.0:
            raise ValueError("aggressiveness_mean must be between 0 and 1")
        if self.aggressiveness_concentration <= 0:
            raise ValueError("aggressiveness_concentration must be greater than 0")


class RiotModel(Model):
    """Basic fan neighborhood model on an N x N lattice.

    The model keeps segregation-related settings and riot-specific settings in
    separate dataclasses for clarity. It still accepts keyword arguments for
    Solara re-instantiation and splits them into both parameter groups.
    """

    def __init__(
        self,
        segregation_params: SegregationParams | None = None,
        riot_params: RiotParams | None = None,
        **kwargs,
    ):
        super().__init__()

        segregation_keys = set(SegregationParams.__dataclass_fields__.keys())
        riot_keys = set(RiotParams.__dataclass_fields__.keys())

        if segregation_params is None:
            segregation_filtered = {k: v for k, v in kwargs.items() if k in segregation_keys}
            self.segregation_params = SegregationParams(**segregation_filtered)
        else:
            self.segregation_params = segregation_params

        if riot_params is None:
            riot_filtered = {k: v for k, v in kwargs.items() if k in riot_keys}
            self.riot_params = RiotParams(**riot_filtered)
        else:
            self.riot_params = riot_params

        self.random = random.Random(self.segregation_params.seed)
        self.home_aggressiveness_ratio = self.segregation_params.home_fraction
        self.away_aggressiveness_ratio = 1.0 - self.segregation_params.home_fraction

        self.grid = SingleGrid(
            width=self.segregation_params.N,
            height=self.segregation_params.N,
            torus=self.segregation_params.torus,
        )

        self.fans = []
        self.police = []
        self.moves_this_step = 0
        self.arrests_this_step = 0
        self.total_arrests = 0
        self.in_warmup = True
        self._warmup_entropy_history = []
        self._warmup_entropy_fine_history = []

        self.create_agents()
        self.update_all_agents()

        self.datacollector = DataCollector(
            model_reporters={
                "Happy": lambda m: m.count_happy(),
                "Unhappy": lambda m: m.count_unhappy(),
                "Home": lambda m: m.count_group(FanGroup.HOME),
                "Away": lambda m: m.count_group(FanGroup.AWAY),
                "Average similarity": lambda m: m.average_similarity(),
                "Segregation index": lambda m: m.average_similarity(),
                "Moves": lambda m: m.moves_this_step,
                "Average last move distance": lambda m: m.average_last_move_distance(),
                "Average last move distance (moved fans)": lambda m: m.average_last_move_distance_of_moved_fans(),
                "Police": lambda m: m.count_police(),
                "Fighting fans": lambda m: m.count_fighting_fans(),
                "Arrests this step": lambda m: m.arrests_this_step,
                "Total arrests": lambda m: m.total_arrests,
                "Average aggressiveness": lambda m: m.average_aggressiveness(),
                "Average perceived win probability": lambda m: m.average_perceived_win_probability(),
                "Average perceived arrest probability": lambda m: m.average_perceived_arrest_probability(),
                "Spatial entropy": lambda m: m.spatial_entropy(),
                "Spatial entropy (fine)": lambda m: m.spatial_entropy_fine(),
                "Entropy CV": lambda m: m.entropy_cv(),
                "Entropy CV (fine)": lambda m: m.entropy_cv_fine(),
                "In warmup": lambda m: int(m.in_warmup),
            }
        )

        self.datacollector.collect(self)

    def create_agents(self):
        total_cells = self.segregation_params.N * self.segregation_params.N
        number_of_agents = int(total_cells * self.segregation_params.agent_density)
        number_of_police = int(total_cells * self.riot_params.police_density)
        number_of_home = int(number_of_agents * self.segregation_params.home_fraction)
        number_of_away = number_of_agents - number_of_home

        all_positions = [
            (x, y)
            for x in range(self.segregation_params.N)
            for y in range(self.segregation_params.N)
        ]
        self.random.shuffle(all_positions)

        required_positions = number_of_agents + number_of_police
        if required_positions > len(all_positions):
            raise ValueError("Agent and police densities exceed available grid cells")

        agent_groups = [FanGroup.HOME] * number_of_home + [FanGroup.AWAY] * number_of_away
        self.random.shuffle(agent_groups)

        for pos, group in zip(all_positions[:number_of_agents], agent_groups):
            fan = Fan(self, group)
            self.fans.append(fan)
            self.grid.place_agent(fan, pos)

        police_positions = all_positions[number_of_agents:required_positions]
        for pos in police_positions:
            police = Police(self)
            self.police.append(police)
            self.grid.place_agent(police, pos)

    def sample_fan_aggressiveness(self, group: FanGroup) -> float:
        mean = self.riot_params.aggressiveness_mean
        if mean is None:
            mean = (
                self.home_aggressiveness_ratio
                if group == FanGroup.HOME
                else self.away_aggressiveness_ratio
            )

        # Keep exact boundary values deterministic; otherwise use a beta
        # distribution with the configured mean and concentration.
        if mean <= 0.0:
            return 0.0
        if mean >= 1.0:
            return 1.0

        concentration = self.riot_params.aggressiveness_concentration
        alpha = mean * concentration
        beta = (1.0 - mean) * concentration
        return self.random.betavariate(alpha, beta)

    def update_all_agents(self):
        for fan in self.fans:
            fan.decide_happiness()
            fan.update_perceived_probabilities()
            fan.fighting = False
        if not self.in_warmup and self.riot_params.fighting_enabled:
            for fan in self.fans:
                fan.decide_fighting()

    def step(self):
        self.moves_this_step = 0
        self.arrests_this_step = 0

        if self.in_warmup:
            # Warmup phase: Schelling movement only, no fighting.
            agents = self.fans[:]
            self.random.shuffle(agents)
            for fan in agents:
                fan.move_if_unhappy()
            self.update_all_agents()

            self._warmup_entropy_history.append(self.spatial_entropy())
            self._warmup_entropy_fine_history.append(self.spatial_entropy_fine())

            fine_window = self._warmup_entropy_fine_history[-self.segregation_params.warmup_window:]
            if self.moves_this_step == 0:
                self.in_warmup = False
            elif len(fine_window) >= self.segregation_params.warmup_window:
                if self._cv(fine_window) < self.segregation_params.warmup_cv_threshold:
                    self.in_warmup = False
        else:
            agents = self.fans[:]
            self.random.shuffle(agents)
            for fan in agents:
                fan.step()

            police_agents = self.police[:]
            self.random.shuffle(police_agents)
            for agent in police_agents:
                agent.step()

            self.update_all_agents()

            self._warmup_entropy_history.append(self.spatial_entropy())
            self._warmup_entropy_fine_history.append(self.spatial_entropy_fine())

        self.datacollector.collect(self)

    def run_model(self, steps=None):
        if steps is None:
            steps = self.segregation_params.steps
        for _ in range(steps):
            self.step()


    def count_group(self, group):
        return sum(fan.group == group for fan in self.fans)

    def count_happy(self):
        return sum(fan.happy for fan in self.fans)

    def count_unhappy(self):
        return len(self.fans) - self.count_happy()

    def count_police(self):
        return len(self.police)

    def count_fighting_fans(self):
        return sum(fan.fighting for fan in self.fans)

    def average_similarity(self):
        if not self.fans:
            return 0.0
        return sum(fan.same_fraction for fan in self.fans) / len(self.fans)
    
    def _zone_entropy(self, zone_size: int) -> float:
        N = self.segregation_params.N
        n_zones_per_side = N // zone_size
        total_entropy = 0.0
        n_zones = 0
        for zone_x in range(n_zones_per_side):
            for zone_y in range(n_zones_per_side):
                home = 0
                away = 0
                for x in range(zone_x * zone_size, (zone_x + 1) * zone_size):
                    for y in range(zone_y * zone_size, (zone_y + 1) * zone_size):
                        agent = self.grid[x][y]
                        if isinstance(agent, Fan):
                            if agent.group == FanGroup.HOME:
                                home += 1
                            else:
                                away += 1
                total = home + away
                if total == 0:
                    continue
                p_home = home / total
                p_away = away / total
                zone_entropy = 0.0
                if p_home > 0:
                    zone_entropy -= p_home * math.log(p_home)
                if p_away > 0:
                    zone_entropy -= p_away * math.log(p_away)
                total_entropy += zone_entropy
                n_zones += 1
        return total_entropy / n_zones if n_zones > 0 else 0.0

    def spatial_entropy(self):
        return self._zone_entropy(self.segregation_params.zone_size)

    def spatial_entropy_fine(self):
        return self._zone_entropy(self.segregation_params.zone_size_fine)

    def _cv(self, window: list) -> float:
        if len(window) < 2:
            return 0.0
        mean = sum(window) / len(window)
        if mean == 0:
            return 0.0
        std = math.sqrt(sum((x - mean) ** 2 for x in window) / len(window))
        return std / mean

    def entropy_cv(self):
        window = self._warmup_entropy_history[-self.segregation_params.warmup_window:]
        return self._cv(window)

    def entropy_cv_fine(self):
        window = self._warmup_entropy_fine_history[-self.segregation_params.warmup_window:]
        return self._cv(window)

    def average_last_move_distance(self):
        if not self.fans:
            return 0.0
        return sum(fan.last_move_distance for fan in self.fans) / len(self.fans)

    def average_last_move_distance_of_moved_fans(self):
        moved_fans = [fan for fan in self.fans if fan.last_move_distance > 0]
        if not moved_fans:
            return 0.0
        return sum(fan.last_move_distance for fan in moved_fans) / len(moved_fans)

    def average_aggressiveness(self):
        if not self.fans:
            return 0.0
        return sum(fan.aggressiveness for fan in self.fans) / len(self.fans)

    def average_perceived_win_probability(self):
        if not self.fans:
            return 0.0
        return sum(fan.perceived_win_probability for fan in self.fans) / len(self.fans)

    def average_perceived_arrest_probability(self):
        if not self.fans:
            return 0.0
        return sum(fan.perceived_arrest_probability for fan in self.fans) / len(self.fans)

    @property
    def params(self):
        """Backward-compatible alias for the segregation parameter set."""
        return self.segregation_params


def export_default_params():
    """Export default parameters for use in the Solara app."""
    segregation_defaults = SegregationParams()
    riot_defaults = RiotParams()
    return {**segregation_defaults.__dict__, **riot_defaults.__dict__}


# Backward-compatible aliases while the branch is in transition.
Group = FanGroup
Household = Fan
RiotModelParams = SegregationParams
SegregationModel = RiotModel
