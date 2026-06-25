"""
Names: Max, Ruben, Mark and Yara
Date: 25-6-2026

Description: This file contains the fan class and all its related logic. VECTORIZED VERSION.
"""

import math
from enum import Enum


class FanGroup(Enum):
    HOME = "home"
    AWAY = "away"


class HawkDoveStrategy(Enum):
    NASH_ESS = "nash_ess"  # deterministic: p = V/C
    LOGIT_PRIOR = "logit_prior"  # logit QRE assuming opponent plays hawk with q=0.5
    LOGIT_QRE = "logit_qre"  # logit QRE: q estimated from local proportion, C stays


class Fan:
    """One fan in the local neighbourhood model."""

    def __init__(self, model, group: FanGroup, is_respawn: bool = False):
        self.model = model
        self.random = model.random
        self.pos = None
        self.group = group
        self.is_respawn = bool(is_respawn)
        self.aggressiveness = self.model.sample_fan_aggressiveness(self.group)
        self.same_fraction = 0.0
        self.happy = False
        self.fighting = False
        self.fight_want = 0.0
        self.perceived_win_probability = 0.0
        self.perceived_arrest_probability = 0.0
        self.last_move_distance = 0
        self.last_opponent_play = "dove"
        self.friend_fans = 0
        self.enemy_fans = 0

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
        return max(dx, dy)

    def calculate_same_fraction(self, neighbors):
        same = sum(
            isinstance(agent, Fan) and agent.group == self.group for agent in neighbors
        )
        denominator = (
            8
            if self.model.segregation_params.count_empty_as_different
            else len(neighbors)
        )
        return 1.0 if denominator == 0 else same / denominator

    def decide_happiness(self, neighbors=None):
        if neighbors is None:
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False, radius=1
            )
        self.same_fraction = self.calculate_same_fraction(neighbors)
        self.happy = (
            self.same_fraction >= self.model.segregation_params.similarity_threshold
        )
        return self.happy

    def set_happiness_from_counts(self, same_count, total_agents):
        """Set same_fraction / happy from precomputed radius-1 neighbour counts.

        Mirrors ``calculate_same_fraction`` exactly: denominator is 8 when empty
        cells count as different, otherwise the number of agents (fans AND
        police) in the Moore-1 neighbourhood.
        """
        if self.model.segregation_params.count_empty_as_different:
            denominator = 8
        else:
            denominator = total_agents
        self.same_fraction = 1.0 if denominator == 0 else same_count / denominator
        self.happy = (
            self.same_fraction >= self.model.segregation_params.similarity_threshold
        )
        return self.happy

    def _hawk_dove_play(self, opponent):
        strategy = self.model.riot_params.hawk_dove_strategy
        cost = self.model.riot_params.hawk_dove_C
        beta = self.model.riot_params.logit_beta

        if strategy == HawkDoveStrategy.NASH_ESS:
            # Pure Nash mixed strategy: p = V/C where V = same-team neighbours
            p_hawk = min(1.0, self.friend_fans / cost) if cost > 0 else 1.0
            return "hawk" if self.random.random() < p_hawk else "dove"

        if strategy == HawkDoveStrategy.LOGIT_PRIOR:
            # Logit QRE with uniform prior: assumes opponent plays hawk with q=0.5
            # ΔE = V/2 - 0.5*C/2 = friend_fans/2 - C/4
            delta_e = self.friend_fans / 2 - cost / 4
            p_hawk = 1 / (1 + math.exp(-beta * delta_e))
            return "hawk" if self.random.random() < p_hawk else "dove"

        if strategy == HawkDoveStrategy.LOGIT_QRE:
            # Logit QRE: q estimated from local proportion, C preserved in delta_e
            total = self.friend_fans + self.enemy_fans
            q = self.enemy_fans / total if total > 0 else 0.5
            delta_e = self.friend_fans / 2 - q * cost / 2
            p_hawk = 1 / (1 + math.exp(-beta * delta_e))
            return "hawk" if self.random.random() < p_hawk else "dove"

        return "hawk"

    def decide_fighting(self, neighbors=None):
        if neighbors is None:
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False, radius=1
            )
        opposing_fans = [
            agent
            for agent in neighbors
            if isinstance(agent, Fan) and agent.group != self.group
        ]

        self.fight_want = self.aggressiveness * self.perceived_win_probability
        fight_margin = self.fight_want - self.perceived_arrest_probability

        if not opposing_fans or fight_margin <= self.model.riot_params.fight_threshold:
            return self.fighting

        opponent = self.random.choice(opposing_fans)
        my_play = self._hawk_dove_play(opponent)
        opponent_play = opponent._hawk_dove_play(self)
        self.last_opponent_play = opponent_play
        opponent.last_opponent_play = my_play

        if my_play == "hawk":
            self.fighting = True
        if opponent_play == "hawk":
            opponent.fighting = True
        return self.fighting

    def set_perceived_from_counts(self, friend, enemy, cops):
        """Set perceived probabilities from precomputed neighbour counts.

        Single source of truth for the perception formula; both the grid-based
        ``update_perceived_probabilities`` and the model's vectorized batch path
        feed into this.
        """
        self.friend_fans = friend
        self.enemy_fans = enemy
        friends_including_self = friend + 1
        total_fans_including_self = friend + enemy + 1
        k = self.model.riot_params.perception_k
        self.perceived_win_probability = math.exp(-k * (enemy / friends_including_self))
        self.perceived_arrest_probability = 1 - math.exp(
            -k * (5 * cops / total_fans_including_self)
        )
        return (
            self.perceived_win_probability,
            self.perceived_arrest_probability,
        )

    def update_perceived_probabilities(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.riot_params.fan_vision,
        )

        enemy_fans = 0
        friend_fans = 0
        cops_in_view = 0
        for agent in neighbors:
            if isinstance(agent, Fan):
                if agent.group == self.group:
                    friend_fans += 1
                else:
                    enemy_fans += 1
            elif getattr(agent, "is_police", False):
                cops_in_view += 1

        return self.set_perceived_from_counts(friend_fans, enemy_fans, cops_in_view)

    def nearest_empty_position(self):
        # Enumerate only the radius-5 window around the agent and keep the empty
        # cells, instead of scanning every empty cell on the grid. The Chebyshev
        # distance is the offset itself, so no per-cell distance call is needed.
        grid = self.model.grid
        N = grid.width
        torus = self.model.segregation_params.torus
        decay = self.model.segregation_params.movement_decay
        x, y = self.pos

        positions = []
        weights = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if torus:
                    nx %= N
                    ny %= N
                elif nx < 0 or ny < 0 or nx >= N or ny >= N:
                    continue
                pos = (nx, ny)
                if grid.is_cell_empty(pos):
                    positions.append(pos)
                    weights.append(math.exp(-decay * max(abs(dx), abs(dy))))

        if not positions:
            # Rare: no empty cell within radius 5 -> fall back to a global scan.
            empty_positions = list(grid.empties)
            if not empty_positions:
                return None
            weights = [
                math.exp(-decay * self.torus_distance(pos)) for pos in empty_positions
            ]
            return self.random.choices(empty_positions, weights=weights, k=1)[0]

        return self.random.choices(positions, weights=weights, k=1)[0]

    def move_if_unhappy(self):
        self.last_move_distance = 0
        if self.decide_happiness():
            if self.random.random() >= self.model.segregation_params.random_move_chance:
                return False
        new_pos = self.nearest_empty_position()
        if new_pos is None:
            return False
        self.last_move_distance = self.torus_distance(new_pos)
        self.model.grid.move_agent(self, new_pos)
        self.model.moves_this_step += 1
        return True

    def step(self):
        self.move_if_unhappy()
