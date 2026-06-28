import math
import random
from enum import Enum
from dataclasses import dataclass

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid


class CitizenState(Enum):
    QUIET = "quiet"
    ACTIVE = "active"


@dataclass
class CivilViolenceParams:
    N: int = 40
    citizen_density: float = 0.70
    cop_density: float = 0.074
    legitimacy: float = 0.80
    threshold: float = 0.10
    citizen_vision: int = 7
    cop_vision: int = 7
    k: float = -math.log(0.1)
    steps: int = 200
    seed: int = 42


class Citizen:
    def __init__(self, model):
        self.model = model
        self.random = model.random
        self.pos = None
        self.hardship = self.model.random.random()
        self.risk_aversion = self.model.random.random()

        self.state = CitizenState.QUIET

        self.grievance = self.calculate_grievance()
        self.arrest_probability = 0.0
        self.net_risk = 0.0

    def calculate_grievance(self):
        L = self.model.params.legitimacy
        return self.hardship * (1 - L)

    def estimate_arrest_probability(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.citizen_vision,
        )

        cops = sum(isinstance(agent, Cop) for agent in neighbors)

        active_citizens = sum(
            isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE
            for agent in neighbors
        )

        # De agent telt zichzelf mee, zoals in de slides.
        active_citizens += 1

        cop_to_active_ratio = cops / active_citizens

        P = 1 - math.exp(-self.model.params.k * cop_to_active_ratio)

        return P

    def calculate_net_risk(self):
        R = self.risk_aversion
        P = self.arrest_probability
        return R * P

    def decide_state(self):
        self.grievance = self.calculate_grievance()
        self.arrest_probability = self.estimate_arrest_probability()
        self.net_risk = self.calculate_net_risk()

        T = self.model.params.threshold

        if self.grievance - self.net_risk > T:
            self.state = CitizenState.ACTIVE
        else:
            self.state = CitizenState.QUIET

    def move_randomly_within_vision(self):
        possible_positions = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.citizen_vision,
        )

        empty_positions = [
            pos for pos in possible_positions if self.model.grid.is_cell_empty(pos)
        ]

        if empty_positions:
            new_pos = self.model.random.choice(empty_positions)
            self.model.grid.move_agent(self, new_pos)

    def step(self):
        self.decide_state()
        self.move_randomly_within_vision()


class Cop:
    def __init__(self, model):
        self.model = model
        self.random = model.random
        self.pos = None

    def find_active_citizens_in_vision(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.cop_vision,
        )

        active_citizens = [
            agent
            for agent in neighbors
            if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE
        ]

        return active_citizens

    def arrest(self, citizen):
        citizen_position = citizen.pos

        self.model.grid.remove_agent(citizen)

        if citizen in self.model.citizens:
            self.model.citizens.remove(citizen)
            self.model.arrested_citizens.append(citizen)

        self.model.grid.move_agent(self, citizen_position)

    def move_randomly_within_vision(self):
        possible_positions = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.cop_vision,
        )

        empty_positions = [
            pos for pos in possible_positions if self.model.grid.is_cell_empty(pos)
        ]

        if empty_positions:
            new_pos = self.model.random.choice(empty_positions)
            self.model.grid.move_agent(self, new_pos)

    def step(self):
        active_citizens = self.find_active_citizens_in_vision()

        if active_citizens:
            citizen = self.model.random.choice(active_citizens)
            self.arrest(citizen)
        else:
            self.move_randomly_within_vision()


class CivilViolenceModel(Model):
    def __init__(self, params: CivilViolenceParams | None = None, **kwargs):
        """Initialize the model.

        Accept either a `CivilViolenceParams` instance via `params`, or individual
        parameters as keyword arguments (e.g. `N=40`, `citizen_density=0.7`). This
        makes the model compatible with Solara's parameter UI which supplies
        keyword args for re-instantiation.
        """
        super().__init__()

        if params is None:
            # Build params from kwargs (or use defaults from dataclass).
            # Filter out any keys that are not fields of CivilViolenceParams
            valid_keys = set(CivilViolenceParams.__dataclass_fields__.keys())
            filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
            self.params = CivilViolenceParams(**filtered)
        else:
            self.params = params

        self.random = random.Random(self.params.seed)

        self.grid = SingleGrid(
            width=self.params.N,
            height=self.params.N,
            torus=True,
        )

        self.citizens = []
        self.cops = []
        self.arrested_citizens = []

        self.create_agents()

        self.datacollector = DataCollector(
            model_reporters={
                "Quiet": lambda m: m.count_citizens(CitizenState.QUIET),
                "Active": lambda m: m.count_citizens(CitizenState.ACTIVE),
                "Cops": lambda m: len(m.cops),
                "Average grievance": lambda m: m.average_grievance(),
                "Average arrest probability": lambda m: m.average_arrest_probability(),
                "Average net risk": lambda m: m.average_net_risk(),
            }
        )

        self.datacollector.collect(self)

    def create_agents(self):
        total_cells = self.params.N * self.params.N
        number_of_citizens = int(total_cells * self.params.citizen_density)
        number_of_cops = int(total_cells * self.params.cop_density)

        all_positions = [
            (x, y) for x in range(self.params.N) for y in range(self.params.N)
        ]

        self.random.shuffle(all_positions)

        citizen_positions = all_positions[:number_of_citizens]
        cop_positions = all_positions[
            number_of_citizens : number_of_citizens + number_of_cops
        ]

        for pos in citizen_positions:
            citizen = Citizen(self)
            self.citizens.append(citizen)
            self.grid.place_agent(citizen, pos)

        for pos in cop_positions:
            cop = Cop(self)
            self.cops.append(cop)
            self.grid.place_agent(cop, pos)

    def step(self):
        all_agents = self.citizens + self.cops
        self.random.shuffle(all_agents)

        for agent in all_agents:
            if isinstance(agent, Citizen) and agent not in self.citizens:
                continue
            agent.step()

        self.datacollector.collect(self)

    def run_model(self, steps=None):
        if steps is None:
            steps = self.params.steps

        for _ in range(steps):
            self.step()

    def count_citizens(self, state):
        return sum(citizen.state == state for citizen in self.citizens)

    def average_grievance(self):
        return sum(c.grievance for c in self.citizens) / len(self.citizens)

    def average_arrest_probability(self):
        if not self.citizens:
            return 0

        return sum(c.arrest_probability for c in self.citizens) / len(self.citizens)

    def average_net_risk(self):
        if not self.citizens:
            return 0

        return sum(c.net_risk for c in self.citizens) / len(self.citizens)
