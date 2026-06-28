"""
DATE: 28-6-2026
NAMES: Ruben, Mark, Yara, Max

Description: Police agent for the riot model.
Disclaimer: AI may be used in with creating the code. We checked the code on functionality, logic and correctness. We are responsible for the code and its content.
"""

import math

try:
    from riot_model.fan import Fan, FanGroup
except ImportError:
    from fan import Fan, FanGroup


## SMALLEST CLASS!
class Police:
    """Police agent that pursues and arrests fighting fans."""

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
        weights = [
            math.exp(-self.model.segregation_params.movement_decay * distance)
            for distance in distances
        ]
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
        best_pos = min(
            empty_positions,
            key=lambda pos: self._grid_distance(pos, target_pos),
        )
        self.last_move_distance = self.torus_distance(best_pos)
        self.model.grid.move_agent(self, best_pos)
        self.model.moves_this_step += 1
        return True

    def arrest(self, fan):
        self.model.grid.remove_agent(fan)
        self.model.fans.remove(fan)
        self.model.arrests_this_step += 1
        self.model.total_arrests += 1

        empty_positions = list(self.model.grid.empties)
        if empty_positions:
            new_pos = self.model.random.choice(empty_positions)
            group = FanGroup.HOME if self.model.random.random() < 0.5 else FanGroup.AWAY
            new_fan = Fan(self.model, group)
            self.model.fans.append(new_fan)
            self.model.grid.place_agent(new_fan, new_pos)
            new_fan.update_perceived_probabilities()

    def step(self):
        self.last_move_distance = 0
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.riot_params.police_vision,
        )
        fighting_fans = [agent for agent in neighbors if isinstance(agent, Fan) and agent.fighting]
        if not fighting_fans:
            self.move()
            return

        target = min(
            fighting_fans,
            key=lambda fan: self.torus_distance(fan.pos),
        )
        if self.torus_distance(target.pos) <= 1:
            self.arrest(target)
        else:
            self.move_toward(target.pos)
