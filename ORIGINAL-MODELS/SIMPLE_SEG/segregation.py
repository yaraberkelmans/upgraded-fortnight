import random
from enum import Enum
from dataclasses import dataclass

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid


class Group(Enum):
    RED = "red"
    GREEN = "green"


@dataclass
class SegregationParams:
    N: int = 40
    agent_density: float = 0.80
    red_fraction: float = 0.50
    similarity_threshold: float = 0.30
    steps: int = 100
    seed: int = 42
    torus: bool = True
    count_empty_as_different: bool = True


class Household:
    """One household in the Schelling segregation model."""

    def __init__(self, model, group: Group):
        self.model = model
        self.random = model.random
        self.pos = None
        self.group = group
        self.same_fraction = 0.0
        self.happy = False

    def calculate_same_fraction(self):
        """
        Calculate the fraction of neighbours that are the same group.

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
            isinstance(agent, Household) and agent.group == self.group for agent in neighbors
        )

        if self.model.params.count_empty_as_different:
            denominator = 8
        else:
            denominator = len(neighbors)

        if denominator == 0:
            return 1.0

        return same / denominator

    def decide_happiness(self):
        self.same_fraction = self.calculate_same_fraction()
        self.happy = self.same_fraction >= self.model.params.similarity_threshold
        return self.happy

    def nearest_empty_position(self):
        empty_positions = list(self.model.grid.empties)
        if not empty_positions:
            return None

        current_x, current_y = self.pos
        width = self.model.grid.width
        height = self.model.grid.height

        def torus_distance(pos):
            x, y = pos
            dx = abs(x - current_x)
            dy = abs(y - current_y)
            if self.model.params.torus:
                dx = min(dx, width - dx)
                dy = min(dy, height - dy)
            # Chebyshev distance matches movement on a Moore grid.
            return max(dx, dy)

        min_distance = min(torus_distance(pos) for pos in empty_positions)
        nearest_positions = [pos for pos in empty_positions if torus_distance(pos) == min_distance]
        return self.random.choice(nearest_positions)

    def move_if_unhappy(self):
        if self.decide_happiness():
            return False

        new_pos = self.nearest_empty_position()
        if new_pos is None:
            return False

        self.model.grid.move_agent(self, new_pos)
        self.model.moves_this_step += 1
        self.decide_happiness()
        return True

    def step(self):
        self.move_if_unhappy()


class SegregationModel(Model):
    """Basic Schelling segregation model on an N x N lattice.

    Accept either a `SegregationParams` instance via `params`, or individual
    parameters as keyword arguments (e.g. `N=40`, `agent_density=0.8`). This
    makes the model compatible with Solara's parameter UI which supplies
    keyword args for re-instantiation.
    """

    def __init__(self, params: SegregationParams | None = None, **kwargs):
        super().__init__()

        if params is None:
            valid_keys = set(SegregationParams.__dataclass_fields__.keys())
            filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
            self.params = SegregationParams(**filtered)
        else:
            self.params = params

        self.random = random.Random(self.params.seed)

        self.grid = SingleGrid(
            width=self.params.N,
            height=self.params.N,
            torus=self.params.torus,
        )

        self.households = []
        self.moves_this_step = 0

        self.create_agents()

        self.datacollector = DataCollector(
            model_reporters={
                "Happy": lambda m: m.count_happy(),
                "Unhappy": lambda m: m.count_unhappy(),
                "Red": lambda m: m.count_group(Group.RED),
                "Green": lambda m: m.count_group(Group.GREEN),
                "Average similarity": lambda m: m.average_similarity(),
                "Segregation index": lambda m: m.average_similarity(),
                "Moves": lambda m: m.moves_this_step,
            }
        )

        self.update_all_happiness()
        self.datacollector.collect(self)

    def create_agents(self):
        total_cells = self.params.N * self.params.N
        number_of_agents = int(total_cells * self.params.agent_density)
        number_of_red = int(number_of_agents * self.params.red_fraction)
        number_of_green = number_of_agents - number_of_red

        all_positions = [(x, y) for x in range(self.params.N) for y in range(self.params.N)]
        self.random.shuffle(all_positions)

        agent_groups = [Group.RED] * number_of_red + [Group.GREEN] * number_of_green
        self.random.shuffle(agent_groups)

        for pos, group in zip(all_positions[:number_of_agents], agent_groups):
            household = Household(self, group)
            self.households.append(household)
            self.grid.place_agent(household, pos)

    def update_all_happiness(self):
        for household in self.households:
            household.decide_happiness()

    def step(self):
        self.moves_this_step = 0
        agents = self.households[:]
        self.random.shuffle(agents)

        for agent in agents:
            agent.step()

        self.update_all_happiness()
        self.datacollector.collect(self)

    def run_model(self, steps=None):
        if steps is None:
            steps = self.params.steps

        for _ in range(steps):
            self.step()
            # Stop early if the city is stable.
            if self.moves_this_step == 0:
                break

    def count_group(self, group):
        return sum(household.group == group for household in self.households)

    def count_happy(self):
        return sum(household.happy for household in self.households)

    def count_unhappy(self):
        return len(self.households) - self.count_happy()

    def average_similarity(self):
        if not self.households:
            return 0.0
        return sum(h.same_fraction for h in self.households) / len(self.households)
