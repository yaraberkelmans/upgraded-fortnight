# ============================================================
# Epstein Civil Violence Model
# Gebaseerd op de lecture slides:
# Classic Models - Civil Violence (Joshua M. Epstein 2002)
# ============================================================

# Als Mesa nog niet geïnstalleerd is:
# %pip install -U mesa pandas matplotlib

import math
import random
from enum import Enum
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid

# import display
from IPython.display import display

# ============================================================
# 1. States van burgers
# ============================================================

class CitizenState(Enum):
    QUIET = "quiet"
    ACTIVE = "active"
    JAILED = "jailed"


# ============================================================
# 2. Parameters
# ============================================================

@dataclass
class CivilViolenceParams:
    # Lattice: N x N grid
    N: int = 40

    # Dichtheden
    citizen_density: float = 0.70
    cop_density: float = 0.074

    # Politieke parameters
    legitimacy: float = 0.80       # L
    threshold: float = 0.10        # T

    # Zicht
    citizen_vision: int = 7        # v voor burgers
    cop_vision: int = 7            # v* voor cops

    # Arrestatiekans
    # k wordt zo gekozen dat P ongeveer 0.9 is wanneer C/A = 1
    k: float = -math.log(0.1)

    # Jail
    max_jail_term: int = 30        # J
    jail_alpha: float = 0.0        # alpha; 0 betekent N = R * P

    # Simulatie
    steps: int = 200
    seed: int = 42


# ============================================================
# 3. Burger-agent
# ============================================================

class Citizen:
    """
    Citizen / activist.

    Volgens de slides:
    - Agents hebben hardship H.
    - Legitimacy L is een globale parameter.
    - Grievance: G = H(1 - L).
    - Agents hebben risk aversion R.
    - Agents schatten arrestatiekans P.
    - Net risk: N = R * P, of N = R * P * J^alpha.
    - Regel:
        if G - N > T:
            active
        else:
            quiet
    """

    def __init__(self, model):
        self.model = model
        self.pos = None

        # H uit U(0,1)
        self.hardship = self.model.random.random()

        # R uit U(0,1)
        self.risk_aversion = self.model.random.random()

        self.state = CitizenState.QUIET
        self.jail_timer = 0

        self.grievance = self.calculate_grievance()
        self.arrest_probability = 0.0
        self.net_risk = 0.0

    def calculate_grievance(self):
        L = self.model.params.legitimacy
        return self.hardship * (1 - L)

    def estimate_arrest_probability(self):
        """
        P = 1 - exp(-k * (C/A)_v)

        C = aantal cops binnen vision
        A = aantal actieve burgers binnen vision

        A >= 1, want de burger telt zichzelf mee als potentieel actieve burger.
        """

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.citizen_vision
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
        """
        N = R * P
        of
        N = R * P * J^alpha

        In de basisspecificatie is alpha = 0,
        waardoor het model terugvalt op N = R * P.
        """

        R = self.risk_aversion
        P = self.arrest_probability
        J = self.model.params.max_jail_term
        alpha = self.model.params.jail_alpha

        if alpha == 0:
            return R * P

        return R * P * (J ** alpha)

    def decide_state(self):
        """
        Beslisregel uit de slides:

        if G - N > T:
            active
        else:
            quiet
        """

        self.grievance = self.calculate_grievance()
        self.arrest_probability = self.estimate_arrest_probability()
        self.net_risk = self.calculate_net_risk()

        T = self.model.params.threshold

        if self.grievance - self.net_risk > T:
            self.state = CitizenState.ACTIVE
        else:
            self.state = CitizenState.QUIET

    def move_randomly_within_vision(self):
        """
        Agents bewegen willekeurig binnen hun vision range,
        zolang daar een lege plek is.
        """

        possible_positions = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.citizen_vision
        )

        empty_positions = [
            pos for pos in possible_positions
            if self.model.grid.is_cell_empty(pos)
        ]

        if empty_positions:
            new_pos = self.model.random.choice(empty_positions)
            self.model.grid.move_agent(self, new_pos)

    def step(self):
        """
        Eén stap voor een burger.
        """

        # Als burger in jail zit, is hij verwijderd van de grid.
        if self.state == CitizenState.JAILED:
            self.jail_timer -= 1

            if self.jail_timer <= 0:
                self.state = CitizenState.QUIET
                self.model.release_from_jail(self)

            return

        # Eerst beslissen of de burger actief wordt.
        self.decide_state()

        # Daarna bewegen.
        self.move_randomly_within_vision()


# ============================================================
# 4. Cop-agent
# ============================================================

class Cop:
    """
    Cop-agent.

    Volgens de slides:
    - Cops hebben vision v*.
    - Cops arresteren actieve agents binnen hun vision.
    - Na arrestatie beweegt de cop naar de plek waar de agent stond.
    """

    def __init__(self, model):
        self.model = model
        self.pos = None

    def find_active_citizens_in_vision(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.cop_vision
        )

        active_citizens = [
            agent for agent in neighbors
            if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE
        ]

        return active_citizens

    def arrest(self, citizen):
        """
        Arresteer een actieve burger.

        De burger wordt tijdelijk uit de omgeving verwijderd.
        De cop beweegt naar de plek van de gearresteerde burger.
        """

        citizen_position = citizen.pos
        cop_position = self.pos

        # Haal burger van de grid.
        self.model.grid.remove_agent(citizen)

        citizen.state = CitizenState.JAILED
        citizen.jail_timer = self.model.random.randint(
            1,
            self.model.params.max_jail_term
        )

        # Cop gaat naar de plek van de gearresteerde burger.
        self.model.grid.move_agent(self, citizen_position)

    def move_randomly_within_vision(self):
        """
        Als de cop niemand arresteert, beweegt hij willekeurig binnen vision.
        """

        possible_positions = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.cop_vision
        )

        empty_positions = [
            pos for pos in possible_positions
            if self.model.grid.is_cell_empty(pos)
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


# ============================================================
# 5. Model
# ============================================================

class CivilViolenceModel(Model):
    """
    Volledig Epstein Civil Violence model volgens de slides.

    Het systeem:
    - N x N lattice
    - Citizens / activists
    - Cops
    - Lokale observatie
    - Heterogene hardship en risk aversion
    - Arrestatie en jail
    """

    def __init__(self, params=CivilViolenceParams()):
        super().__init__()

        self.params = params
        self.random = random.Random(params.seed)

        self.grid = SingleGrid(
            width=params.N,
            height=params.N,
            torus=True
        )

        self.citizens = []
        self.cops = []
        self.jailed_citizens = []

        self.create_agents()

        self.datacollector = DataCollector(
            model_reporters={
                "Quiet": lambda m: m.count_citizens(CitizenState.QUIET),
                "Active": lambda m: m.count_citizens(CitizenState.ACTIVE),
                "Jailed": lambda m: m.count_citizens(CitizenState.JAILED),
                "Cops": lambda m: len(m.cops),
                "Average grievance": lambda m: m.average_grievance(),
                "Average arrest probability": lambda m: m.average_arrest_probability(),
                "Average net risk": lambda m: m.average_net_risk(),
            }
        )

        self.datacollector.collect(self)

    def create_agents(self):
        """
        Vul het N x N lattice met citizens, cops en lege plekken.
        """

        total_cells = self.params.N * self.params.N
        number_of_citizens = int(total_cells * self.params.citizen_density)
        number_of_cops = int(total_cells * self.params.cop_density)

        all_positions = [
            (x, y)
            for x in range(self.params.N)
            for y in range(self.params.N)
        ]

        self.random.shuffle(all_positions)

        citizen_positions = all_positions[:number_of_citizens]
        cop_positions = all_positions[
            number_of_citizens:number_of_citizens + number_of_cops
        ]

        for pos in citizen_positions:
            citizen = Citizen(self)
            self.citizens.append(citizen)
            self.grid.place_agent(citizen, pos)

        for pos in cop_positions:
            cop = Cop(self)
            self.cops.append(cop)
            self.grid.place_agent(cop, pos)

    def release_from_jail(self, citizen):
        """
        Zet een burger terug op een willekeurige lege plek.
        """

        empty_cells = list(self.grid.empties)

        if empty_cells:
            new_pos = self.random.choice(empty_cells)
            self.grid.place_agent(citizen, new_pos)
        else:
            # Als er geen plek is, blijft de burger nog even buiten het systeem.
            citizen.state = CitizenState.JAILED
            citizen.jail_timer = 1

    def step(self):
        """
        Eén volledige modelstap.

        Belangrijk:
        - Burgers en cops worden in willekeurige volgorde geactiveerd.
        - Dit past bij ABM-logica: lokale interacties veroorzaken globale patronen.
        """

        all_agents = self.citizens + self.cops
        self.random.shuffle(all_agents)

        for agent in all_agents:
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
        free_citizens = [
            c for c in self.citizens
            if c.state != CitizenState.JAILED
        ]

        if not free_citizens:
            return 0

        return sum(c.arrest_probability for c in free_citizens) / len(free_citizens)

    def average_net_risk(self):
        free_citizens = [
            c for c in self.citizens
            if c.state != CitizenState.JAILED
        ]

        if not free_citizens:
            return 0

        return sum(c.net_risk for c in free_citizens) / len(free_citizens)


# ============================================================
# 6. Visualisatie van grid
# ============================================================

def plot_grid(model, title="Civil Violence grid"):
    quiet_x, quiet_y = [], []
    active_x, active_y = [], []
    jailed_x, jailed_y = [], []
    cop_x, cop_y = [], []

    for citizen in model.citizens:
        if citizen.state == CitizenState.JAILED:
            continue

        x, y = citizen.pos

        if citizen.state == CitizenState.QUIET:
            quiet_x.append(x)
            quiet_y.append(y)

        elif citizen.state == CitizenState.ACTIVE:
            active_x.append(x)
            active_y.append(y)

    for cop in model.cops:
        x, y = cop.pos
        cop_x.append(x)
        cop_y.append(y)

    plt.figure(figsize=(7, 7))

    plt.scatter(quiet_x, quiet_y, s=20, label="Quiet citizens")
    plt.scatter(active_x, active_y, s=20, label="Active citizens")
    plt.scatter(cop_x, cop_y, s=30, label="Cops")

    plt.xlim(-1, model.params.N)
    plt.ylim(-1, model.params.N)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.legend()
    plt.show()


# ============================================================
# 7. Resultaten plotten
# ============================================================

def plot_results(results):
    ax = results[["Quiet", "Active", "Jailed"]].plot(figsize=(10, 5))

    ax.set_title("Epstein Civil Violence Model")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of citizens")

    plt.show()


def plot_risk_results(results):
    ax = results[
        ["Average grievance", "Average arrest probability", "Average net risk"]
    ].plot(figsize=(10, 5))

    ax.set_title("Average grievance, arrest probability and net risk")
    ax.set_xlabel("Step")
    ax.set_ylabel("Average value")

    plt.show()


# ============================================================
# 8. Model runnen
# ============================================================

params = CivilViolenceParams(
    N=40,

    citizen_density=0.70,
    cop_density=0.004,

    legitimacy=0.80,
    threshold=0.10,

    citizen_vision=7,
    cop_vision=7,

    k=-math.log(0.1),

    max_jail_term=30,
    jail_alpha=0.0,

    steps=200,
    seed=42
)

model = CivilViolenceModel(params)

plot_grid(model, title="Starttoestand")

model.run_model()

results = model.datacollector.get_model_vars_dataframe()

display(results.head())
display(results.tail())

plot_results(results)
plot_risk_results(results)

plot_grid(model, title="Eindtoestand")


# ============================================================
# 9. Experiment: lage versus hoge legitimiteit
# ============================================================

def run_legitimacy_experiment(legitimacy, steps=200, seed=42):
    experiment_params = CivilViolenceParams(
        N=40,

        citizen_density=0.70,
        cop_density=0.074,

        legitimacy=legitimacy,
        threshold=0.10,

        citizen_vision=7,
        cop_vision=7,

        k=-math.log(0.1),

        max_jail_term=30,
        jail_alpha=0.0,

        steps=steps,
        seed=seed
    )

    experiment_model = CivilViolenceModel(experiment_params)
    experiment_model.run_model()

    df = experiment_model.datacollector.get_model_vars_dataframe()
    df["Legitimacy"] = legitimacy

    return df


df_high_legitimacy = run_legitimacy_experiment(
    legitimacy=0.85,
    steps=200,
    seed=42
)

df_low_legitimacy = run_legitimacy_experiment(
    legitimacy=0.65,
    steps=200,
    seed=42
)

plt.figure(figsize=(10, 5))

plt.plot(
    df_high_legitimacy.index,
    df_high_legitimacy["Active"],
    label="Legitimacy = 0.85"
)

plt.plot(
    df_low_legitimacy.index,
    df_low_legitimacy["Active"],
    label="Legitimacy = 0.65"
)

plt.title("Effect van legitimacy op active citizens")
plt.xlabel("Step")
plt.ylabel("Number of active citizens")
plt.legend()
plt.show()