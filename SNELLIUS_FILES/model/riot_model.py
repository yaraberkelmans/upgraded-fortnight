"""
Names: Max, Ruben, Mark and Yara
Date: 25-6-2026

Description: This file contains the riot model class and all its related logic.
    This version is vectorized so it perfoms better than the original riot model but the code is more complex. Resulting in less readablity.
    The code is partially vectorized using AI meaning that the seperation of functions inside the file is not the best. We made the choices however what and how to do it.
"""

import random
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid

from .fan import Fan, FanGroup, HawkDoveStrategy
from .police import Police


def _box_sum(plane, radius, wrap):
    # Make a box-sum of the given 2D array, summing all neighbors within the given chebyshevs radius.
    mode = "wrap" if wrap else "constant"
    N = plane.shape[0]
    padded = np.pad(plane, radius, mode=mode)
    total = np.zeros_like(plane)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            total += padded[radius + dx : radius + dx + N, radius + dy : radius + dy + N]
    return total


@dataclass
class SegregationParams:
    N: int = 40
    agent_density: float = 0.80
    home_fraction: float = 0.50
    similarity_threshold: float = 0.30
    movement_decay: float = 1.0
    steps: int = 100
    seed: int = 42
    torus: bool = True
    count_empty_as_different: bool = True
    zone_size: int = 10
    zone_size_fine: int = 4
    warmup_cv_threshold: float = 0.05
    warmup_window: int = 10
    random_move_chance: float = 0.005
    collect_data: bool = True


@dataclass
class RiotParams:
    police_density: float = 0.05
    perception_k: float = 0.693
    fan_vision: int = 2
    fight_threshold: float = 0.0
    police_vision: int = 5
    logit_beta: float = 5.0
    hawk_dove_strategy: HawkDoveStrategy = HawkDoveStrategy.LOGIT_PRIOR
    hawk_dove_C: float = 4.0
    aggressiveness_mean: float | None = None
    aggressiveness_concentration: float = 12.0
    fighting_enabled: bool = True

    def __post_init__(self):
        if isinstance(self.hawk_dove_strategy, str):
            self.hawk_dove_strategy = HawkDoveStrategy(self.hawk_dove_strategy)
        if self.aggressiveness_mean is not None and not 0.0 <= self.aggressiveness_mean <= 1.0:
            raise ValueError("aggressiveness_mean must be between 0 and 1")
        if self.aggressiveness_concentration <= 0:
            raise ValueError("aggressiveness_concentration must be greater than 0")


class RiotModel(Model):
    """Basic fan neighborhood model on an N x N lattice.

    The model keeps segregation-related settings and riot-specific settings in
    separate dataclasses for clarity. It still accepts keyword arguments for
    Solara re-instantiation and splits them into both parameter groups.
    """

    def __init__(
        self,
        segregation_params: SegregationParams | None = None,
        riot_params: RiotParams | None = None,
        **kwargs,
    ):
        super().__init__()

        segregation_keys = set(SegregationParams.__dataclass_fields__.keys())
        riot_keys = set(RiotParams.__dataclass_fields__.keys())

        if segregation_params is None:
            segregation_filtered = {k: v for k, v in kwargs.items() if k in segregation_keys}
            self.segregation_params = SegregationParams(**segregation_filtered)
        else:
            self.segregation_params = segregation_params

        if riot_params is None:
            riot_filtered = {k: v for k, v in kwargs.items() if k in riot_keys}
            self.riot_params = RiotParams(**riot_filtered)
        else:
            self.riot_params = riot_params

        self.random = random.Random(self.segregation_params.seed)
        self.home_aggressiveness_ratio = 1.0 - self.segregation_params.home_fraction
        self.away_aggressiveness_ratio = self.segregation_params.home_fraction

        self.grid = SingleGrid(
            width=self.segregation_params.N,
            height=self.segregation_params.N,
            torus=self.segregation_params.torus,
        )

        self.fans = []
        self.police = []
        self.moves_this_step = 0
        self.arrests_this_step = 0
        self.arrested_fans_this_step = []
        self.total_arrests = 0
        self.in_warmup = True
        self._last_entropy_fine = 0.0
        self._warmup_entropy_fine_history = deque(maxlen=self.segregation_params.warmup_window)

        # Cached aggregate statistics, refreshed every tick by
        # _build_spatial_state(); the datacollector reporters read these.
        self._home_occ = None
        self._away_occ = None
        self._home1 = None
        self._away1 = None
        self._stat_n_fans = 0
        self._stat_happy = 0
        self._stat_home = 0
        self._stat_away = 0
        self._stat_fighting = 0
        self._stat_sum_same = 0.0
        self._stat_sum_pwin = 0.0
        self._stat_sum_parr = 0.0
        self._stat_sum_aggr = 0.0
        self._stat_sum_lmd = 0.0
        self._stat_moved = 0

        self.create_agents()
        self.update_all_agents()

        self.datacollector = None
        # MODEL REPORTER CAN BE TURNED OFF REALLY NICE FOR PERFORMANCE :)
        if self.segregation_params.collect_data:
            self.datacollector = DataCollector(
                model_reporters={
                    "Happy": lambda m: m.count_happy(),
                    "Unhappy": lambda m: m.count_unhappy(),
                    "Home": lambda m: m.count_group(FanGroup.HOME),
                    "Away": lambda m: m.count_group(FanGroup.AWAY),
                    "Average similarity": lambda m: m.average_similarity(),
                    "Segregation index": lambda m: m.average_similarity(),
                    "Moves": lambda m: m.moves_this_step,
                    "Average last move distance": lambda m: m.average_last_move_distance(),
                    "Average last move distance (moved fans)": lambda m: m.average_last_move_distance_of_moved_fans(),
                    "Police": lambda m: m.count_police(),
                    "Fighting fans": lambda m: m.count_fighting_fans(),
                    "Arrests this step": lambda m: m.arrests_this_step,
                    "Total arrests": lambda m: m.total_arrests,
                    "Average aggressiveness": lambda m: m.average_aggressiveness(),
                    "Average perceived win probability": lambda m: m.average_perceived_win_probability(),
                    "Average perceived arrest probability": lambda m: m.average_perceived_arrest_probability(),
                    "Spatial entropy (fine)": lambda m: m._last_entropy_fine,
                    "Entropy CV (fine)": lambda m: m.entropy_cv_fine(),
                    "In warmup": lambda m: int(m.in_warmup),
                }
            )
            self.datacollector.collect(self)

    def create_agents(self):
        total_cells = self.segregation_params.N * self.segregation_params.N
        number_of_agents = int(total_cells * self.segregation_params.agent_density)
        number_of_police = int(total_cells * self.riot_params.police_density)
        number_of_home = int(number_of_agents * self.segregation_params.home_fraction)
        number_of_away = number_of_agents - number_of_home

        all_positions = [
            (x, y)
            for x in range(self.segregation_params.N)
            for y in range(self.segregation_params.N)
        ]
        self.random.shuffle(all_positions)

        required_positions = number_of_agents + number_of_police
        if required_positions > len(all_positions):
            raise ValueError("Agent and police densities exceed available grid cells")

        agent_groups = [FanGroup.HOME] * number_of_home + [FanGroup.AWAY] * number_of_away
        self.random.shuffle(agent_groups)

        for pos, group in zip(all_positions[:number_of_agents], agent_groups):
            fan = Fan(self, group)
            self.fans.append(fan)
            self.grid.place_agent(fan, pos)

        police_positions = all_positions[number_of_agents:required_positions]
        for pos in police_positions:
            police = Police(self)
            self.police.append(police)
            self.grid.place_agent(police, pos)

    def sample_fan_aggressiveness(self, group: FanGroup) -> float:
        mean = self.riot_params.aggressiveness_mean
        if mean is None:
            mean = (
                self.home_aggressiveness_ratio
                if group == FanGroup.HOME
                else self.away_aggressiveness_ratio
            )

        # Keep exact boundary values deterministic; otherwise use a beta
        # distribution with the configured mean and concentration.
        if mean <= 0.0:
            return 0.0
        if mean >= 1.0:
            return 1.0

        concentration = self.riot_params.aggressiveness_concentration
        alpha = mean * concentration
        beta = (1.0 - mean) * concentration
        return self.random.betavariate(alpha, beta)

    def _build_spatial_state(self):
        """Rebuild the spatial planes and refresh per-fan + aggregate state.

        Runs once per tick after all movement/arrests have settled. Computes
        neighbour counts via vectorized box-sums, writes perceived
        probabilities and happiness back onto each fan, and caches the
        aggregate statistics consumed by the datacollector. Does NOT touch the
        RNG, so trajectories stay identical to the grid-based model.
        """
        N = self.segregation_params.N
        wrap = self.segregation_params.torus

        home = np.zeros((N, N), dtype=np.int64)
        away = np.zeros((N, N), dtype=np.int64)
        police = np.zeros((N, N), dtype=np.int64)
        for fan in self.fans:
            x, y = fan.pos
            if fan.group == FanGroup.HOME:
                home[x, y] = 1
            else:
                away[x, y] = 1
        for cop in self.police:
            x, y = cop.pos
            police[x, y] = 1

        self._home_occ = home
        self._away_occ = away

        fan_vision = self.riot_params.fan_vision
        home_v = _box_sum(home, fan_vision, wrap)
        away_v = _box_sum(away, fan_vision, wrap)
        cop_v = _box_sum(police, fan_vision, wrap)
        home1 = _box_sum(home, 1, wrap)
        away1 = _box_sum(away, 1, wrap)
        police1 = _box_sum(police, 1, wrap)
        self._home1 = home1
        self._away1 = away1

        stat_happy = 0
        sum_same = 0.0
        sum_pwin = 0.0
        sum_parr = 0.0
        sum_aggr = 0.0
        sum_lmd = 0.0
        moved = 0
        n_home = 0
        n_away = 0

        for fan in self.fans:
            x, y = fan.pos
            if fan.group == FanGroup.HOME:
                friend = int(home_v[x, y])
                enemy = int(away_v[x, y])
                same1 = int(home1[x, y])
                n_home += 1
            else:
                friend = int(away_v[x, y])
                enemy = int(home_v[x, y])
                same1 = int(away1[x, y])
                n_away += 1
            cops = int(cop_v[x, y])
            total_agents1 = int(home1[x, y] + away1[x, y] + police1[x, y])

            fan.set_perceived_from_counts(friend, enemy, cops)
            fan.set_happiness_from_counts(same1, total_agents1)
            fan.fighting = False

            stat_happy += fan.happy
            sum_same += fan.same_fraction
            sum_pwin += fan.perceived_win_probability
            sum_parr += fan.perceived_arrest_probability
            sum_aggr += fan.aggressiveness
            lmd = fan.last_move_distance
            sum_lmd += lmd
            if lmd > 0:
                moved += 1

        self._stat_n_fans = len(self.fans)
        self._stat_happy = stat_happy
        self._stat_home = n_home
        self._stat_away = n_away
        self._stat_sum_same = sum_same
        self._stat_sum_pwin = sum_pwin
        self._stat_sum_parr = sum_parr
        self._stat_sum_aggr = sum_aggr
        self._stat_sum_lmd = sum_lmd
        self._stat_moved = moved
        self._stat_fighting = 0

    def update_all_agents(self):
        self._build_spatial_state()

        if not self.in_warmup and self.riot_params.fighting_enabled:
            threshold = self.riot_params.fight_threshold
            for fan in self.fans:
                # Same early-return predicate as Fan.decide_fighting, evaluated
                # cheaply from the spatial planes so we only pay for a radius-1
                # get_neighbors when this fan can actually fight this tick.
                fan.fight_want = fan.aggressiveness * fan.perceived_win_probability
                margin = fan.fight_want - fan.perceived_arrest_probability
                x, y = fan.pos
                enemy1 = (
                    int(self._away1[x, y]) if fan.group == FanGroup.HOME else int(self._home1[x, y])
                )
                if enemy1 == 0 or margin <= threshold:
                    continue
                neighbors = self.grid.get_neighbors(
                    fan.pos, moore=True, include_center=False, radius=1
                )
                fan.decide_fighting(neighbors)

            self._stat_fighting = sum(fan.fighting for fan in self.fans)

    def step(self):
        self.moves_this_step = 0
        self.arrests_this_step = 0
        self.arrested_fans_this_step = []

        if self.in_warmup:
            # Warmup phase: Schelling movement only, no fighting.
            agents = self.fans[:]
            self.random.shuffle(agents)
            for fan in agents:
                fan.move_if_unhappy()
            self.update_all_agents()

            self._last_entropy_fine = self.spatial_entropy_fine()
            self._warmup_entropy_fine_history.append(self._last_entropy_fine)

            fine_window = self._warmup_entropy_fine_history
            if self.moves_this_step == 0:
                self.in_warmup = False
            elif len(fine_window) >= self.segregation_params.warmup_window:
                if self._cv(fine_window) < self.segregation_params.warmup_cv_threshold:
                    self.in_warmup = False
        else:
            agents = self.fans[:]
            self.random.shuffle(agents)
            for fan in agents:
                fan.step()

            police_agents = self.police[:]
            self.random.shuffle(police_agents)
            for agent in police_agents:
                agent.step()

            self.update_all_agents()

            # Entropy/CV are warm-up diagnostics. Do not keep extending the
            # warm-up history once the riot phase has started.
            if self.segregation_params.collect_data:
                self._last_entropy_fine = self.spatial_entropy_fine()

        if self.datacollector is not None:
            self.datacollector.collect(self)

    def run_model(self, steps=None):
        if steps is None:
            steps = self.segregation_params.steps
        for _ in range(steps):
            self.step()

    def count_group(self, group):
        return self._stat_home if group == FanGroup.HOME else self._stat_away

    def count_happy(self):
        return self._stat_happy

    def count_unhappy(self):
        return self._stat_n_fans - self._stat_happy

    def count_police(self):
        return len(self.police)

    def count_fighting_fans(self):
        return self._stat_fighting

    def average_similarity(self):
        if not self._stat_n_fans:
            return 0.0
        return self._stat_sum_same / self._stat_n_fans

    def _zone_entropy(self, zone_size: int) -> float:
        # Reuses the occupancy planes built in _build_spatial_state(); cast to
        # float32 and using the identical expressions keeps this bit-for-bit
        # equal to the grid-based model (entropy feeds the warmup-exit CV check,
        # so it must match exactly).
        N = self.segregation_params.N
        n_zones = N // zone_size

        home = self._home_occ.astype(np.float32)
        away = self._away_occ.astype(np.float32)

        home_z = home.reshape(n_zones, zone_size, n_zones, zone_size).sum(axis=(1, 3))
        away_z = away.reshape(n_zones, zone_size, n_zones, zone_size).sum(axis=(1, 3))
        total_z = home_z + away_z

        mask = total_z > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            p_home = np.where(mask, home_z / np.where(mask, total_z, 1), 0.0)
            p_away = np.where(mask, away_z / np.where(mask, total_z, 1), 0.0)

            entropy = -np.where(p_home > 0, p_home * np.log(p_home), 0.0) - np.where(
                p_away > 0, p_away * np.log(p_away), 0.0
            )

        n_nonempty = mask.sum()
        return float(entropy[mask].sum() / n_nonempty) if n_nonempty > 0 else 0.0

    def spatial_entropy_fine(self):
        return self._zone_entropy(self.segregation_params.zone_size_fine)

    def _cv(self, window: list) -> float:
        if len(window) < 2:
            return 0.0
        mean = sum(window) / len(window)
        if mean == 0:
            return 0.0
        std = math.sqrt(sum((x - mean) ** 2 for x in window) / len(window))
        return std / mean

    def entropy_cv_fine(self):
        return self._cv(self._warmup_entropy_fine_history)

    def average_last_move_distance(self):
        if not self._stat_n_fans:
            return 0.0
        return self._stat_sum_lmd / self._stat_n_fans

    def average_last_move_distance_of_moved_fans(self):
        if not self._stat_moved:
            return 0.0
        return self._stat_sum_lmd / self._stat_moved

    def average_aggressiveness(self):
        if not self._stat_n_fans:
            return 0.0
        return self._stat_sum_aggr / self._stat_n_fans

    def average_perceived_win_probability(self):
        if not self._stat_n_fans:
            return 0.0
        return self._stat_sum_pwin / self._stat_n_fans

    def average_perceived_arrest_probability(self):
        if not self._stat_n_fans:
            return 0.0
        return self._stat_sum_parr / self._stat_n_fans

    @property
    def params(self):
        """Backward-compatible alias for the segregation parameter set."""
        return self.segregation_params


def export_default_params():
    """Export default parameters for use in the Solara app."""
    segregation_defaults = SegregationParams()
    riot_defaults = RiotParams()
    return {**segregation_defaults.__dict__, **riot_defaults.__dict__}


# Backward-compatible aliases while the branch is in transition. HELPS US RUN STUFF.
Group = FanGroup
Household = Fan
RiotModelParams = SegregationParams
SegregationModel = RiotModel
