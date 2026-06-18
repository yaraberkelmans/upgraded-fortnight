import random
import math
from enum import Enum
from dataclasses import dataclass

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid


AGGRESSIVENESS_K = 12.0


class FanGroup(Enum):
    HOME = "home"
    AWAY = "away"


class HawkDoveStrategy(Enum):
    NASH_ESS = "nash_ess"
    AGGRESSIVENESS = "aggressiveness"
    BOURGEOIS = "bourgeois"
    ANTI_BOURGEOIS = "anti_bourgeois"
    TIT_FOR_TAT = "tit_for_tat"


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


@dataclass
class RiotParams:
    police_density: float = 0.05
    perception_k: float = 0.693
    fan_vision: int = 2
    fight_threshold: float = 0.0
    police_vision: int = 5
    hawk_dove_strategy: HawkDoveStrategy = HawkDoveStrategy.AGGRESSIVENESS
    hawk_dove_C: float = 4.0

    def __post_init__(self):
        if isinstance(self.hawk_dove_strategy, str):
            self.hawk_dove_strategy = HawkDoveStrategy(self.hawk_dove_strategy)


class Fan:
    """One fan in the local neighborhood model."""

    def __init__(self, model, group: FanGroup):
        self.model = model
        self.random = model.random
        self.pos = None
        self.group = group
        # Sampled once at spawn time so aggressiveness stays fixed for the fan.
        self.aggressiveness = self.model.sample_fan_aggressiveness(self.group)
        self.same_fraction = 0.0
        self.happy = False
        self.fighting = False
        self.fight_want = 0.0
        self.perceived_win_probability = 0.0
        self.perceived_arrest_probability = 0.0
        self.last_move_distance = 0
        self.last_opponent_play = "dove"

    def torus_distance(self, pos):
        current_x, current_y = self.pos
        x, y = pos
        dx = abs(x - current_x)
        dy = abs(y - current_y)
        if self.model.segregation_params.torus:
            width = self.model.grid.width
            height = self.model.grid.height
            dx = min(dx, width - dx)
            dy = min(dy, height - dy)
        # Chebyshev distance matches movement on a Moore grid.
        return max(dx, dy)

    def calculate_same_fraction(self):
        """
        Calculate the fraction of neighbours that are the same fan group.

        The lecture version uses the 8 surrounding cells as the denominator.
        Therefore, by default, empty lots count as not-same. If you prefer the
        common Schelling variant, set count_empty_as_different=False so that
        only occupied neighbouring cells are counted.
        """
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=1,
        )

        same = sum(
            isinstance(agent, Fan) and agent.group == self.group
            for agent in neighbors
        )

        if self.model.segregation_params.count_empty_as_different:
            denominator = 8
        else:
            denominator = len(neighbors)

        if denominator == 0:
            return 1.0

        return same / denominator

    def decide_happiness(self):
        self.same_fraction = self.calculate_same_fraction()
        self.happy = self.same_fraction >= self.model.segregation_params.similarity_threshold
        return self.happy

    def _hawk_dove_play(self, opponent):
        strategy = self.model.riot_params.hawk_dove_strategy
        C = self.model.riot_params.hawk_dove_C

        if strategy == HawkDoveStrategy.NASH_ESS:
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False, radius=1
            )
            V = sum(1 for a in neighbors if isinstance(a, Fan) and a.group == self.group)
            p_hawk = min(1.0, V / C) if C > 0 else 1.0
            return "hawk" if self.random.random() < p_hawk else "dove"

        elif strategy == HawkDoveStrategy.AGGRESSIVENESS:
            return "hawk" if self.random.random() < self.aggressiveness else "dove"

        elif strategy == HawkDoveStrategy.BOURGEOIS:
            return "hawk" if self.group == FanGroup.HOME else "dove"

        elif strategy == HawkDoveStrategy.ANTI_BOURGEOIS:
            return "hawk" if self.group == FanGroup.AWAY else "dove"

        elif strategy == HawkDoveStrategy.TIT_FOR_TAT:
            return opponent.last_opponent_play

        return "hawk"

    def decide_fighting(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=1,
        )

        opposing_fans = [
            agent for agent in neighbors
            if isinstance(agent, Fan) and agent.group != self.group
        ]

        self.fight_want = self.aggressiveness * self.perceived_win_probability
        fight_margin = self.fight_want - self.perceived_arrest_probability

        if not opposing_fans or fight_margin <= self.model.riot_params.fight_threshold:
            return self.fighting

        opponent = self.random.choice(opposing_fans)
        my_play = self._hawk_dove_play(opponent)
        opp_play = opponent._hawk_dove_play(self)

        # Both sides remember what the other played (for TfT)
        self.last_opponent_play = opp_play
        opponent.last_opponent_play = my_play

        # Both participants get their fighting flag updated from this game
        if my_play == "hawk":
            self.fighting = True
        if opp_play == "hawk":
            opponent.fighting = True

        return self.fighting

    def update_perceived_probabilities(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.riot_params.fan_vision,
        )

        enemy_fans = sum(
            isinstance(agent, Fan) and agent.group != self.group
            for agent in neighbors
        )
        friend_fans = sum(
            isinstance(agent, Fan) and agent.group == self.group
            for agent in neighbors
        )
        cops_in_view = sum(getattr(agent, "is_police", False) for agent in neighbors)

        friends_including_self = friend_fans + 1
        total_fans_including_self = friend_fans + enemy_fans + 1
        k = self.model.riot_params.perception_k

        self.perceived_win_probability = math.exp(-k * (enemy_fans / friends_including_self))
        self.perceived_arrest_probability = 1 - math.exp(-k * (5 * cops_in_view / total_fans_including_self))

        return self.perceived_win_probability, self.perceived_arrest_probability

    def nearest_empty_position(self):
        empty_positions = list(self.model.grid.empties)
        if not empty_positions:
            return None

        distances = [self.torus_distance(pos) for pos in empty_positions]
        weights = [math.exp(-self.model.segregation_params.movement_decay * distance) for distance in distances]
        return self.random.choices(empty_positions, weights=weights, k=1)[0]

    def move_if_unhappy(self):
        self.last_move_distance = 0

        if self.decide_happiness():
            return False

        new_pos = self.nearest_empty_position()
        if new_pos is None:
            return False

        self.last_move_distance = self.torus_distance(new_pos)
        self.model.grid.move_agent(self, new_pos)
        self.model.moves_this_step += 1
        self.decide_happiness()
        return True

    def step(self):
        self.move_if_unhappy()
        self.decide_fighting()


class Police:
    """One police agent that pursues and arrests fighting fans."""

    is_police = True

    def __init__(self, model):
        self.model = model
        self.random = model.random
        self.pos = None
        self.last_move_distance = 0

    def _grid_distance(self, pos_a, pos_b):
        dx = abs(pos_a[0] - pos_b[0])
        dy = abs(pos_a[1] - pos_b[1])
        if self.model.segregation_params.torus:
            dx = min(dx, self.model.grid.width - dx)
            dy = min(dy, self.model.grid.height - dy)
        return max(dx, dy)

    def torus_distance(self, pos):
        return self._grid_distance(self.pos, pos)

    def nearest_empty_position(self):
        empty_positions = list(self.model.grid.empties)
        if not empty_positions:
            return None

        distances = [self.torus_distance(pos) for pos in empty_positions]
        weights = [math.exp(-self.model.segregation_params.movement_decay * distance) for distance in distances]
        return self.random.choices(empty_positions, weights=weights, k=1)[0]

    def move(self):
        self.last_move_distance = 0

        new_pos = self.nearest_empty_position()
        if new_pos is None:
            return False

        self.last_move_distance = self.torus_distance(new_pos)
        self.model.grid.move_agent(self, new_pos)
        self.model.moves_this_step += 1
        return True

    def move_toward(self, target_pos):
        empty_positions = list(self.model.grid.empties)
        if not empty_positions:
            return False

        best_pos = min(empty_positions, key=lambda p: self._grid_distance(p, target_pos))
        self.last_move_distance = self.torus_distance(best_pos)
        self.model.grid.move_agent(self, best_pos)
        self.model.moves_this_step += 1
        return True

    def arrest(self, fan):
        self.model.grid.remove_agent(fan)
        self.model.fans.remove(fan)
        self.model.arrests_this_step += 1

    def step(self):
        self.last_move_distance = 0

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.riot_params.police_vision,
        )
        fighting_fans = [a for a in neighbors if isinstance(a, Fan) and a.fighting]

        if not fighting_fans:
            self.move()
            return

        target = min(fighting_fans, key=lambda a: self.torus_distance(a.pos))

        if self.torus_distance(target.pos) <= 1:
            self.arrest(target)
        else:
            self.move_toward(target.pos)


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

        self.create_agents()

        self.update_all_happiness()
        self.update_all_perceived_probabilities()
        self.update_all_fighting()

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
                "Average aggressiveness": lambda m: m.average_aggressiveness(),
                "Average perceived win probability": lambda m: m.average_perceived_win_probability(),
                "Average perceived arrest probability": lambda m: m.average_perceived_arrest_probability(),
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
        group_ratio = self.home_aggressiveness_ratio if group == FanGroup.HOME else self.away_aggressiveness_ratio
        alpha = max(1e-6, group_ratio * AGGRESSIVENESS_K)
        beta = max(1e-6, (1.0 - group_ratio) * AGGRESSIVENESS_K)
        return self.random.betavariate(alpha, beta)

    def update_all_happiness(self):
        for fan in self.fans:
            fan.decide_happiness()

    def update_all_fighting(self):
        for fan in self.fans:
            fan.fighting = False
        for fan in self.fans:
            fan.decide_fighting()

    def update_all_perceived_probabilities(self):
        for fan in self.fans:
            fan.update_perceived_probabilities()

    def step(self):
        self.moves_this_step = 0
        self.arrests_this_step = 0
        agents = self.fans[:]
        self.random.shuffle(agents)

        for agent in agents:
            agent.step()

        police_agents = self.police[:]
        self.random.shuffle(police_agents)

        for agent in police_agents:
            agent.step()

        self.update_all_happiness()
        self.update_all_perceived_probabilities()
        self.update_all_fighting()
        self.datacollector.collect(self)

    def run_model(self, steps=None):
        if steps is None:
            steps = self.segregation_params.steps

        for _ in range(steps):
            self.step()
            # Stop early if the city is stable.
            if self.moves_this_step == 0:
                break

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


# Backward-compatible aliases while the branch is in transition.
Group = FanGroup
Household = Fan
RiotModelParams = SegregationParams
SegregationModel = RiotModel
