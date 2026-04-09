"""
JIT Supply Chain Simulator — Environment implementation.

Three tasks of increasing difficulty:
  easy_reorder      : 1 SKU, deterministic demand 10/day, 30-day horizon
  medium_stochastic : 1 SKU, Poisson(λ=10) demand, 50-day horizon
  hard_multi_sku    : 3 SKUs, Poisson demands, shared warehouse capacity, 60-day horizon

Reward structure:
  +1.0  per unit of demand fulfilled
  -0.05 per unit held in retailer stock each day
  -2.0  per unit of unsatisfied demand
  -0.5  per step where a loop/no-op pattern is detected
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from ..models import SupplyChainAction, SupplyChainObservation, SupplyChainState
except ImportError:
    from models import SupplyChainAction, SupplyChainObservation, SupplyChainState

from openenv.core import Environment

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy_reorder": {
        "sku_names": ["sku_0"],
        "demand_type": "deterministic",
        "demand_mean": {"sku_0": 10},
        "lead_time": {"sku_0": 2},
        "warehouse_capacity": 9999,
        "max_days": 30,
        "initial_stock": {"sku_0": 20},
        "holding_cost": 0.05,
        "stockout_cost": 2.0,
        "fulfill_reward": 1.0,
    },
    "medium_stochastic": {
        "sku_names": ["sku_0"],
        "demand_type": "poisson",
        "demand_mean": {"sku_0": 10},
        "lead_time": {"sku_0": 3},
        "warehouse_capacity": 9999,
        "max_days": 50,
        "initial_stock": {"sku_0": 30},
        "holding_cost": 0.05,
        "stockout_cost": 2.0,
        "fulfill_reward": 1.0,
    },
    "hard_multi_sku": {
        "sku_names": ["sku_0", "sku_1", "sku_2"],
        "demand_type": "poisson",
        "demand_mean": {"sku_0": 8, "sku_1": 12, "sku_2": 6},
        "lead_time": {"sku_0": 2, "sku_1": 3, "sku_2": 4},
        "warehouse_capacity": 120,
        "max_days": 60,
        "initial_stock": {"sku_0": 20, "sku_1": 25, "sku_2": 15},
        "holding_cost": 0.05,
        "stockout_cost": 2.0,
        "fulfill_reward": 1.0,
    },
}


class SupplyChainEnvironment(Environment):
    """JIT Supply Chain Simulator."""

    def __init__(self, task: str = "easy_reorder", seed: Optional[int] = None):
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS)}")
        self.task = task
        self.cfg = TASK_CONFIGS[task]
        self.rng = np.random.default_rng(seed)
        self._state: Optional[SupplyChainState] = None
        self._prev_orders: Optional[Dict[str, int]] = None

    def _sample_demand(self, sku: str) -> int:
        mean = self.cfg["demand_mean"][sku]
        if self.cfg["demand_type"] == "deterministic":
            return int(mean)
        return int(self.rng.poisson(mean))

    def reset(self, **kwargs) -> SupplyChainObservation:
        # Allow task to be overridden at reset time
        task = kwargs.get("task", self.task)
        if task != self.task:
            if task not in TASK_CONFIGS:
                raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS)}")
            self.task = task
            self.cfg = TASK_CONFIGS[task]

        cfg = self.cfg
        self._prev_orders = None

        pipeline: Dict[str, List[int]] = {
            sku: [0] * cfg["lead_time"][sku]
            for sku in cfg["sku_names"]
        }

        self._state = SupplyChainState(
            task=self.task,
            day=0,
            max_days=cfg["max_days"],
            sku_names=cfg["sku_names"],
            retailer_stock={sku: cfg["initial_stock"][sku] for sku in cfg["sku_names"]},
            warehouse_stock={sku: 9999 for sku in cfg["sku_names"]},
            pipeline=pipeline,
            cumulative_reward=0.0,
            cumulative_holding_cost=0.0,
            cumulative_stockout_cost=0.0,
            done=False,
            episode_id=str(uuid.uuid4())[:8],
            step_count=0,
        )
        return self._make_obs(demand={sku: 0 for sku in cfg["sku_names"]})

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        s = self._state
        if s is None or s.done:
            raise RuntimeError("Call reset() before step().")

        cfg = self.cfg
        orders = action.orders
        reward = 0.0
        messages = []

        # 1. Deliver pipeline orders arriving today
        for sku in cfg["sku_names"]:
            if s.pipeline[sku]:
                arriving = s.pipeline[sku].pop(0)
                s.retailer_stock[sku] = min(
                    s.retailer_stock[sku] + arriving,
                    cfg["warehouse_capacity"],
                )
                if arriving > 0:
                    messages.append(f"{arriving} units of {sku} delivered.")

        # 2. Satisfy demand
        demand_realised: Dict[str, int] = {}
        for sku in cfg["sku_names"]:
            d = self._sample_demand(sku)
            demand_realised[sku] = d
            fulfilled = min(d, s.retailer_stock[sku])
            shortfall = d - fulfilled
            s.retailer_stock[sku] -= fulfilled

            reward += cfg["fulfill_reward"] * fulfilled
            stockout_penalty = cfg["stockout_cost"] * shortfall
            reward -= stockout_penalty
            s.cumulative_stockout_cost += stockout_penalty

            if shortfall > 0:
                messages.append(f"STOCKOUT {sku}: {shortfall} unmet.")

        # 3. Holding cost
        for sku in cfg["sku_names"]:
            holding = cfg["holding_cost"] * s.retailer_stock[sku]
            reward -= holding
            s.cumulative_holding_cost += holding

        # 4. Place new orders
        loop_detected = False
        for sku in cfg["sku_names"]:
            qty = max(0, int(orders.get(sku, 0)))
            if self._prev_orders is not None:
                if qty == self._prev_orders.get(sku, 0) and s.retailer_stock[sku] == 0:
                    loop_detected = True
            s.pipeline[sku].append(qty)
            if qty > 0:
                messages.append(f"Ordered {qty} units of {sku}.")

        if loop_detected:
            reward -= 0.5
            messages.append("Loop detected — penalty applied.")

        self._prev_orders = dict(orders)

        # 5. Advance day
        s.day += 1
        s.step_count += 1
        s.cumulative_reward += reward
        s.done = s.day >= cfg["max_days"]

        return self._make_obs(
            demand=demand_realised,
            reward=reward,
            message=" | ".join(messages) if messages else "All clear.",
        )

    @property
    def state(self) -> SupplyChainState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def grade(self) -> float:
        """
        Produce a normalised score [0, 1] based on episode performance.
        Score = clamp((cumulative_reward - worst_case) / (best_case - worst_case), 0, 1)
        """
        if self._state is None:
            return 0.0
        s = self._state
        cfg = self.cfg
        max_days = cfg["max_days"]

        # Best case: fulfill all expected demand with minimal holding
        best_reward = sum(
            cfg["fulfill_reward"] * cfg["demand_mean"][sku]
            - cfg["holding_cost"] * max(cfg["demand_mean"][sku], 1)
            for sku in cfg["sku_names"]
        ) * max_days

        # Worst case: every unit is a stockout
        worst_reward = sum(
            -cfg["stockout_cost"] * cfg["demand_mean"][sku]
            for sku in cfg["sku_names"]
        ) * max_days

        if best_reward <= worst_reward:
            return 0.5

        score = (s.cumulative_reward - worst_reward) / (best_reward - worst_reward)
        return float(max(0.0, min(1.0, score)))

    def _make_obs(
        self,
        demand: Dict[str, int],
        reward: float = 0.0,
        message: str = "",
    ) -> SupplyChainObservation:
        s = self._state
        return SupplyChainObservation(
            day=s.day,
            retailer_stock=dict(s.retailer_stock),
            pipeline={sku: sum(s.pipeline[sku]) for sku in s.sku_names},
            demand=demand,
            reward=reward,
            cumulative_reward=s.cumulative_reward,
            message=message,
            done=s.done,
            task=s.task,
        )
