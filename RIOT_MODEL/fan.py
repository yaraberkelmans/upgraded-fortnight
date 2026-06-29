"""
DATE: 28-6-2026
NAMES: Ruben, Mark, Yara, Max

Description: Fan agent for the riot model. Contains the Fan class, which represents a fan in the simulation. The Fan class includes methods for deciding happiness, fighting behavior, and movement based on local neighborhood conditions and model parameters.
Disclaimer: AI may be used in with creating the code. We checked the code on functionality, logic and correctness. We are responsible for the code and its content.
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

    def __init__(self, model, group: FanGroup):
        self.model = model
        self.random = model.random
        self.pos = None
        self.group = group
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

    def calculate_same_fraction(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=1
        )
        same = sum(isinstance(agent, Fan) and agent.group == self.group for agent in neighbors)

        denominator = (
            8 if self.model.segregation_params.count_empty_as_different else len(neighbors)
        )
        return 1.0 if denominator == 0 else same / denominator

    def decide_happiness(self):
        self.same_fraction = self.calculate_same_fraction()
        self.happy = self.same_fraction >= self.model.segregation_params.similarity_threshold
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

    def decide_fighting(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=1
        )
        opposing_fans = [
            agent for agent in neighbors if isinstance(agent, Fan) and agent.group != self.group
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

    def update_perceived_probabilities(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.model.riot_params.fan_vision,
        )

        self.enemy_fans = 0
        self.friend_fans = 0
        cops_in_view = 0
        for agent in neighbors:
            if isinstance(agent, Fan):
                if agent.group == self.group:
                    self.friend_fans += 1
                else:
                    self.enemy_fans += 1
            elif getattr(agent, "is_police", False):
                cops_in_view += 1

        friends_including_self = self.friend_fans + 1
        total_fans_including_self = self.friend_fans + self.enemy_fans + 1
        k = self.model.riot_params.perception_k
        self.perceived_win_probability = math.exp(-k * (self.enemy_fans / friends_including_self))
        self.perceived_arrest_probability = 1 - math.exp(
            -k * (5 * cops_in_view / total_fans_including_self)
        )
        return (
            self.perceived_win_probability,
            self.perceived_arrest_probability,
        )

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
        self.decide_happiness()
        return True

    def step(self):
        self.move_if_unhappy()
