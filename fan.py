"""Fan agents for the mixed football-riot model.

This file intentionally contains only fan-related concepts. Police behaviour is
kept in cop.py and model-level setup/measurement is kept in mixed_models.py.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Avoid runtime circular imports.
    from mixed_models import MixedFootballRiotsModel


class FanGroup(Enum):
    """The two rival supporter groups."""

    HOME = "home"
    AWAY = "away"


class FanState(Enum):
    """Current behavioural state of a fan."""

    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"


class Fan:
    """A supporter agent.

    A fan combines the Civil Violence idea of grievance/risk with the Schelling
    idea of local group composition. The fan becomes aggressive only when there
    is at least one rival nearby and the local aggression score is above the
    model threshold.
    """

    is_fan = True
    is_police = False

    def __init__(self, model: "MixedFootballRiotsModel", group: FanGroup):
        self.model = model
        self.random = model.random
        self.pos = None
        self.group = group

        # Individual heterogeneity, comparable to hardship/risk aversion in the
        # Civil Violence model.
        self.hostility = self.random.random()
        self.risk_aversion = self.random.random()

        self.state = FanState.PASSIVE
        self.rival_fraction = 0.0
        self.own_fraction = 0.0
        self.police_fraction = 0.0
        self.local_aggressive_fraction = 0.0
        # Group-level baseline aggression. For now every fan in the same group
        # receives the same value, derived from the global group composition.
        # Example: if 30% are AWAY, every AWAY fan gets 0.70 baseline aggression;
        # if 70% are HOME, every HOME fan gets 0.30 baseline aggression.
        self.base_aggression = self.model.group_base_aggression(self.group)
        self.arrest_probability = 0.0
        self.aggression_score = 0.0
        self.same_fraction = 0.0
        self.happy = True

    @property
    def rival_group(self) -> FanGroup:
        return FanGroup.AWAY if self.group == FanGroup.HOME else FanGroup.HOME

    def neighbours(self, radius: int | None = None):
        if radius is None:
            radius = self.model.params.fan_vision
        return self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=radius,
        )

    def local_counts(self):
        neighbours = self.neighbours(self.model.params.fan_vision)
        own = sum(isinstance(agent, Fan) and agent.group == self.group for agent in neighbours)
        rivals = sum(isinstance(agent, Fan) and agent.group != self.group for agent in neighbours)
        police = sum(getattr(agent, "is_police", False) for agent in neighbours)
        return own, rivals, police, len(neighbours)

    def estimate_arrest_probability(self, police: int, aggressive_fans: int) -> float:
        """Estimate arrest probability using the Civil Violence ratio logic."""
        aggressive_fans = max(1, aggressive_fans)
        ratio = police / aggressive_fans
        return 1 - math.exp(-self.model.params.k * ratio)

    def calculate_same_fraction(self) -> float:
        """Schelling-style local similarity around the fan."""
        neighbours = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=1,
        )
        same = sum(isinstance(agent, Fan) and agent.group == self.group for agent in neighbours)
        fan_neighbours = sum(isinstance(agent, Fan) for agent in neighbours)

        if self.model.params.count_empty_as_different:
            denominator = 8
        else:
            denominator = fan_neighbours

        if denominator == 0:
            return 1.0
        return same / denominator

    def decide_happiness(self) -> bool:
        self.same_fraction = self.calculate_same_fraction()
        self.happy = self.same_fraction >= self.model.params.similarity_threshold
        return self.happy

    def decide_state(self) -> None:
        """Choose passive/aggressive behaviour for the current step.

        Rule: no rival nearby means no fan-fan game, so the fan stays passive.
        With rivals nearby, aggression increases through the fan's group-level
        baseline aggression, rival presence, local own-group confidence, and
        local riot contagion. Aggression decreases through police presence and
        perceived arrest risk.
        """
        neighbours = self.neighbours(self.model.params.fan_vision)
        total_neighbours = max(1, len(neighbours))

        own = sum(isinstance(agent, Fan) and agent.group == self.group for agent in neighbours)
        rivals = sum(isinstance(agent, Fan) and agent.group != self.group for agent in neighbours)
        police = sum(getattr(agent, "is_police", False) for agent in neighbours)
        aggressive_nearby = sum(
            isinstance(agent, Fan) and agent.state == FanState.AGGRESSIVE
            for agent in neighbours
        )

        # If the fan was already aggressive, include itself in the arrest-risk
        # denominator. For social contagion we only count neighbours, not self.
        aggressive_for_risk = aggressive_nearby + int(self.state == FanState.AGGRESSIVE)

        self.own_fraction = own / total_neighbours
        self.rival_fraction = rivals / total_neighbours
        self.police_fraction = police / total_neighbours
        self.local_aggressive_fraction = aggressive_nearby / total_neighbours
        self.base_aggression = self.model.group_base_aggression(self.group)
        self.arrest_probability = self.estimate_arrest_probability(police, aggressive_for_risk)

        if rivals == 0:
            self.aggression_score = 0.0
            self.state = FanState.PASSIVE
            return

        p = self.model.params
        risk = self.risk_aversion * self.arrest_probability
        self.aggression_score = (
            p.base_aggression_weight * self.base_aggression
            + p.hostility_weight * self.hostility
            + p.rival_weight * self.rival_fraction
            + p.own_group_weight * self.own_fraction
            + p.riot_contagion_weight * self.local_aggressive_fraction
            - p.police_deterrence * self.police_fraction
            - p.risk_weight * risk
        )

        if self.aggression_score > p.aggression_threshold:
            self.state = FanState.AGGRESSIVE
        else:
            self.state = FanState.PASSIVE

    def candidate_positions(self):
        neighbourhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.params.movement_radius,
        )
        return [pos for pos in neighbourhood if self.model.grid.is_cell_empty(pos)]

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
            return max(dx, dy)

        min_distance = min(torus_distance(pos) for pos in empty_positions)
        nearest = [pos for pos in empty_positions if torus_distance(pos) == min_distance]
        return self.random.choice(nearest)

    def preferred_side_score(self, pos) -> float:
        """Higher means better under separation.

        HOME is encouraged toward the left side, AWAY toward the right side.
        """
        x, _ = pos
        width = max(1, self.model.grid.width - 1)
        normalized_x = x / width
        if self.group == FanGroup.HOME:
            return 1.0 - normalized_x
        return normalized_x

    def distance_to(self, pos) -> int:
        """Chebyshev distance on the model grid, respecting torus if enabled."""
        current_x, current_y = self.pos
        x, y = pos
        dx = abs(x - current_x)
        dy = abs(y - current_y)

        if self.model.params.torus:
            width = self.model.grid.width
            height = self.model.grid.height
            dx = min(dx, width - dx)
            dy = min(dy, height - dy)

        return max(dx, dy)

    def same_fraction_at_position(self, pos) -> float:
        """Schelling score for a possible target position.

        This asks: if this fan moved to `pos`, what fraction of radius-1
        neighbours would be from the same supporter group? Empty cells can count
        as different, matching the original Schelling lecture variant.
        """
        neighbours = self.model.grid.get_neighbors(
            pos,
            moore=True,
            include_center=False,
            radius=1,
        )

        same = sum(isinstance(agent, Fan) and agent.group == self.group for agent in neighbours)
        fan_neighbours = sum(isinstance(agent, Fan) for agent in neighbours)

        if self.model.params.count_empty_as_different:
            denominator = 8
        else:
            denominator = fan_neighbours

        if denominator == 0:
            return 1.0
        return same / denominator

    def schelling_jump_position(self):
        """Choose a relocation target for a spatially unhappy fan.

        The target is not just the nearest empty cell. It is weighted by:
        - exponential distance: nearby moves remain more likely;
        - target similarity: places with more own-group neighbours are preferred;
        - separation side bias: under separation, own side is preferred.
        """
        empty_positions = list(self.model.grid.empties)
        if not empty_positions:
            return None

        p = self.model.params
        candidates = []
        weights = []

        for pos in empty_positions:
            distance = self.distance_to(pos)
            if distance == 0:
                continue
            if p.max_jump_distance is not None and distance > p.max_jump_distance:
                continue

            target_similarity = self.same_fraction_at_position(pos)
            weight = math.exp(-p.jump_decay * distance)
            weight *= 1.0 + p.schelling_weight * target_similarity

            if p.separation_strength > 0:
                weight *= 1.0 + p.separation_strength * self.preferred_side_score(pos)

            candidates.append(pos)
            weights.append(weight)

        if not candidates:
            return None
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def exponential_jump_position(self):
        """Choose any empty cell with probability exp(-decay * distance).

        This keeps local movement most likely, but makes rare longer jumps
        possible. Under separation the weights are additionally biased toward
        the fan's own side of the grid.
        """
        empty_positions = list(self.model.grid.empties)
        if not empty_positions:
            return None

        p = self.model.params
        candidates = []
        weights = []

        for pos in empty_positions:
            distance = self.distance_to(pos)
            if distance == 0:
                continue
            if p.max_jump_distance is not None and distance > p.max_jump_distance:
                continue

            weight = math.exp(-p.jump_decay * distance)

            if p.separation_strength > 0:
                side_bias = 1.0 + p.separation_strength * self.preferred_side_score(pos)
                weight *= side_bias

            candidates.append(pos)
            weights.append(weight)

        if not candidates:
            return None
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def move(self) -> None:
        p = self.model.params

        # Schelling-like relocation: unhappy fans seek an empty position with
        # a better own-group neighbourhood, while still usually moving nearby.
        if p.use_schelling_movement and not self.decide_happiness():
            new_pos = self.schelling_jump_position()
            if new_pos is not None:
                self.model.grid.move_agent(self, new_pos)
                self.model.moves_this_step += 1
                return

        if p.use_exponential_jump:
            new_pos = self.exponential_jump_position()
        else:
            candidates = self.candidate_positions()
            if not candidates:
                return

            # Normalization: free movement. Separation: biased movement to own side.
            if self.random.random() < p.separation_strength:
                best_score = max(self.preferred_side_score(pos) for pos in candidates)
                best_positions = [
                    pos for pos in candidates
                    if self.preferred_side_score(pos) == best_score
                ]
                new_pos = self.random.choice(best_positions)
            else:
                new_pos = self.random.choice(candidates)

        if new_pos is None:
            return

        self.model.grid.move_agent(self, new_pos)
        self.model.moves_this_step += 1

    def step(self) -> None:
        self.decide_state()
        self.move()
