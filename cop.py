"""Police agents for the mixed football-riot model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fan import Fan, FanState

if TYPE_CHECKING:
    from mixed_models import MixedFootballRiotsModel


class Cop:
    """Police agent that targets aggressive fans within vision.

    If no aggressive fan is visible, the cop patrols. Under separation the patrol
    is biased toward the border between the two supporter areas; under
    normalization it is random.
    """

    is_fan = False
    is_police = True

    def __init__(self, model: "MixedFootballRiotsModel"):
        self.model = model
        self.random = model.random
        self.pos = None

    def neighbours(self):
        return self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.cop_vision,
        )

    def find_aggressive_fans_in_vision(self):
        return [
            agent
            for agent in self.neighbours()
            if isinstance(agent, Fan) and agent.state == FanState.AGGRESSIVE
        ]

    def arrest(self, fan: Fan) -> None:
        fan_position = fan.pos
        self.model.grid.remove_agent(fan)

        if fan in self.model.fans:
            self.model.fans.remove(fan)
        self.model.arrested_fans.append(fan)
        self.model.arrests_this_step += 1

        # Move into the arrested fan's position, matching the original Civil
        # Violence implementation.
        self.model.grid.move_agent(self, fan_position)

    def patrol_candidates(self):
        neighbourhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.cop_vision,
        )
        return [pos for pos in neighbourhood if self.model.grid.is_cell_empty(pos)]

    def border_score(self, pos) -> float:
        """Higher is closer to the middle border between supporter zones."""
        x, _ = pos
        middle = (self.model.grid.width - 1) / 2
        max_distance = max(1, middle)
        return 1.0 - abs(x - middle) / max_distance

    def patrol(self) -> None:
        candidates = self.patrol_candidates()
        if not candidates:
            return

        if self.random.random() < self.model.params.separation_strength:
            best_score = max(self.border_score(pos) for pos in candidates)
            best = [pos for pos in candidates if self.border_score(pos) == best_score]
            new_pos = self.random.choice(best)
        else:
            new_pos = self.random.choice(candidates)

        self.model.grid.move_agent(self, new_pos)

    def step(self) -> None:
        aggressive_fans = self.find_aggressive_fans_in_vision()
        if aggressive_fans:
            self.arrest(self.random.choice(aggressive_fans))
        else:
            self.patrol()
