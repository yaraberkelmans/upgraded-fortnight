from mesa import Agent, Model
import random
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule


def portrayal(agent):
    return {"Shape": "circle", "r": 0.5, "Filled": "true", "Color": "#c0392b" if agent.color == "red" else "#2980b9", "Layer": 0}


class Person(Agent):
    def __init__(self, unique_id, model, color):
        super().__init__(unique_id, model)
        self.color = color
        self.happy = False

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=1)
        if neighbors:
            same = sum(1 for agent in neighbors if getattr(agent, "color", None) == self.color)
            self.happy = same / len(neighbors) >= self.model.similarity
        else:
            self.happy = True
        if not self.happy:
            self.model.grid.move_agent(self, random.choice(list(self.model.grid.empties)))


class Epstein(Model):
    def __init__(self, width=20, height=20, density=0.8, similarity=0.5):
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.similarity = similarity
        count = int(width * height * density)
        for i in range(count):
            agent = Person(i, self, "red" if i % 2 == 0 else "blue")
            self.schedule.add(agent)
            self.grid.place_agent(agent, random.choice(list(self.grid.empties)))
        self.datacollector = DataCollector(
            model_reporters={"Happy": lambda model: sum(1 for agent in model.schedule.agents if agent.happy) / len(model.schedule.agents)}
        )

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


grid = CanvasGrid(portrayal, 20, 20, 500, 500)
chart = ChartModule([{"Label": "Happy", "Color": "#2ecc71"}])
server = ModularServer(Epstein, [grid, chart], "Simple Epstein Segregation", {"width": 20, "height": 20, "density": 0.8, "similarity": 0.5})

if __name__ == "__main__":
    server.launch()