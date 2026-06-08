from mesa import Agent, Model
import random
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation


def portrayal(agent):
    if isinstance(agent, Cop):
        return {"Shape": "rect", "w": 0.8, "h": 0.8, "Filled": "true", "Color": "#111111", "Layer": 1}
    color = "#999999" if agent.jailed > 0 else "#d33f49" if agent.active else "#4caf50"
    return {"Shape": "circle", "r": 0.5, "Filled": "true", "Color": color, "Layer": 0}


class Citizen(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.grievance = random.random()
        self.risk = random.random()
        self.jailed = 0
        self.active = False

    def step(self):
        self.model.grid.move_agent(self, random.choice(list(self.model.grid.empties)))
        if self.jailed > 0:
            self.jailed -= 1
            self.active = False
            return
        cops = sum(1 for agent in self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=1) if isinstance(agent, Cop))
        self.active = self.grievance > self.risk and cops < 2


class Cop(Agent):
    def step(self):
        self.model.grid.move_agent(self, random.choice(list(self.model.grid.empties)))
        for agent in self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=1):
            if isinstance(agent, Citizen) and agent.active:
                agent.jailed = self.model.jail_term
                agent.active = False


class CivilViolence(Model):
    def __init__(self, width=20, height=20, citizens=80, cops=10, jail_term=5):
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.jail_term = jail_term
        for i in range(citizens):
            agent = Citizen(i, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, random.choice(list(self.grid.empties)))
        for i in range(cops):
            agent = Cop(citizens + i, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, random.choice(list(self.grid.empties)))
        self.datacollector = DataCollector(
            model_reporters={
                "Active": lambda model: sum(1 for agent in model.schedule.agents if isinstance(agent, Citizen) and agent.active),
                "Jailed": lambda model: sum(1 for agent in model.schedule.agents if isinstance(agent, Citizen) and agent.jailed > 0),
            }
        )

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


if __name__ == "__main__":
    model = CivilViolence(width=20, height=20, citizens=80, cops=10, jail_term=5)
    for step_number in range(1, 51):
        model.step()
        active = sum(1 for agent in model.schedule.agents if isinstance(agent, Citizen) and agent.active)
        jailed = sum(1 for agent in model.schedule.agents if isinstance(agent, Citizen) and agent.jailed > 0)
        print(f"step {step_number:02d}: active={active} jailed={jailed}")