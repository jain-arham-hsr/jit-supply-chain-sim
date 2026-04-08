"""
JIT Supply Chain Simulator — Environment implementation.

Three tasks of increasing difficulty:
  easy_reorder      : 1 SKU, deterministic demand 10/day, 30-day horizon
  medium_stochastic : 1 SKU, Poisson(λ=10) demand, 50-day horizon
  hard_multi_sku    : 3 SKUs, Poisson demands, shared warehouse capacity, 60-day horizon

Reward structure:
  +1.0  per unit of demand fulfilled (incentivises service level)
  -0.05 per unit held in retailer stock each day (holding cost)
  -2.0  per unit of unsatisfied demand (stockout penalty)
  -0.5  per step where a loop/no-op pattern is detected (anti-loop)
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ..models import SupplyChainAction, SupplyChainObservation, SupplyChainState
except ImportError:
    from models import SupplyChainAction, SupplyChainObservation, SupplyChainState

from openenv.core import Environment

# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy_reorder": {
        "sku_names": ["sku_0"],
        "demand_type": "deterministic",
        "demand_mean": {"sku_0": 10},
        "lead_time": {"sku_0": 2},           # fixed 2-day lead time
        "warehouse_capacity": 9999,          # unconstrained
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
        "lead_time": {"sku_0": 3},           # fixed 3-day lead time
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
        "warehouse_capacity": 120,           # shared capacity across all SKUs
        "max_days": 60,
        "initial_stock": {"sku_0": 20, "sku_1": 25, "sku_2": 15},
        "holding_cost": 0.05,
        "stockout_cost": 2.0,
        "fulfill_reward": 1.0,
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupplyChainEnvironment(Environment):
    """
    JIT Supply Chain Simulator.

    The agent receives daily observations of retailer stock, pipeline orders,
    and realised demand, then issues replenishment orders for each SKU.
    """

    def __init__(self, task: str = "easy_reorder", seed: Optional[int] = None):
        if task not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS)}"
            )
        self.task = task
        self.cfg = TASK_CONFIGS[task]
        self.rng = np.random.default_rng(seed)
        self._episode_id: Optional[str] = None
        self._state: Optional[SupplyChainState] = None
        self._prev_orders: Optional[Dict[str, int]] = None   # for loop detection

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_demand(self, sku: str) -> int:
        mean = self.cfg["demand_mean"][sku]
        if self.cfg["demand_type"] == "deterministic":
            return int(mean)
        return int(self.rng.poisson(mean))

    def _total_stock(self, s: SupplyChainState) -> int:
        return sum(s.retailer_stock.values())

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> SupplyChainObservation:
        cfg = self.cfg
        self._episode_id = str(uuid.uuid4())[:8]
        self._prev_orders = None

        # Initialise pipeline as list of zeros per lead-time bucket per SKU
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
            episode_id=self._episode_id,
            step_count=0,
        )
        return self._make_obs(demand={sku: 0 for sku in cfg["sku_names"]})

    def step(
        self, action: SupplyChainAction
    ) -> Tuple[SupplyChainObservation, float, bool, Dict[str, Any]]:
        s = self._state
        if s is None or s.done:
            raise RuntimeError("Call reset() before step().")

        cfg = self.cfg
        orders = action.orders  # Dict[str, int]
        reward = 0.0
        messages = []

        # --- 1. Advance pipeline: deliver orders arriving today ---
        for sku in cfg["sku_names"]:
            if cfg["pipeline"][sku]:
                arriving = s.pipeline[sku].pop(0)
                s.retailer_stock[sku] = min(
                    s.retailer_stock[sku] + arriving,
                    cfg["warehouse_capacity"],
                )
                if arriving > 0:
                    messages.append(f"{arriving} units of {sku} delivered.")

        # --- 2. Satisfy demand ---
        demand_realised: Dict[str, int] = {}
        for sku in cfg["sku_names"]:
            d = self._sample_demand(sku)
            demand_realised[sku] = d
            fulfilled = min(d, s.retailer_stock[sku])
            shortfall = d - fulfilled
            s.retailer_stock[sku] -= fulfilled

            # Reward: fulfilled demand
            reward += cfg["fulfill_reward"] * fulfilled
            # Penalty: stockout
            stockout_penalty = cfg["stockout_cost"] * shortfall
            reward -= stockout_penalty
            s.cumulative_stockout_cost += stockout_penalty

            if shortfall > 0:
                messages.append(
                    f"STOCKOUT {sku}: {shortfall} units unmet (demand={d}, stock={fulfilled+s.retailer_stock[sku]})."
                )

        # --- 3. Holding cost ---
        for sku in cfg["sku_names"]:
            holding = cfg["holding_cost"] * s.retailer_stock[sku]
            reward -= holding
            s.cumulative_holding_cost += holding

        # --- 4. Place new orders into pipeline ---
        loop_detected = False
        for sku in cfg["sku_names"]:
            qty = max(0, int(orders.get(sku, 0)))

            # Anti-loop: penalise if agent keeps repeating same non-zero or
            # zero order when clearly out-of-stock
            if self._prev_orders is not None:
                prev_qty = self._prev_orders.get(sku, 0)
                if qty == prev_qty and s.retailer_stock[sku] == 0:
                    loop_detected = True

            s.pipeline[sku].append(qty)
            if qty > 0:
                messages.append(f"Ordered {qty} units of {sku}.")

        if loop_detected:
            reward -= 0.5
            messages.append("Loop/no-op pattern detected — small penalty applied.")

        self._prev_orders = dict(orders)

        # --- 5. Advance day ---
        s.day += 1
        s.step_count += 1
        s.cumulative_reward += reward
        s.done = s.day >= cfg["max_days"]

        obs = self._make_obs(
            demand=demand_realised,
            message=" | ".join(messages) if messages else "All clear.",
        )
        info = {
            "holding_cost_this_step": cfg["holding_cost"]
            * sum(s.retailer_stock.values()),
            "stockout_this_step": {
                sku: max(0, demand_realised[sku] - s.retailer_stock[sku])
                for sku in cfg["sku_names"]
            },
        }
        return obs, reward, s.done, info

    def state(self) -> SupplyChainState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ------------------------------------------------------------------
    # Graders (0.0 – 1.0 score for each task)
    # ------------------------------------------------------------------

    def grade(self) -> float:
        """
        Produce a normalised score [0, 1] based on episode performance.

        Score = clamp((cumulative_reward - worst_case) /
                      (best_case - worst_case), 0, 1)
        """
        if self._state is None:
            return 0.0
        s = self._state
        cfg = self.cfg
        max_days = cfg["max_days"]

        # Best case: fulfill all expected demand, zero stock-out, minimal holding
        best_reward = sum(
            cfg["fulfill_reward"] * cfg["demand_mean"][sku]
            - cfg["holding_cost"] * max(cfg["demand_mean"][sku], 1)
            for sku in cfg["sku_names"]
        ) * max_days

        # Worst case: every unit is a stock-out
        worst_reward = sum(
            -cfg["stockout_cost"] * cfg["demand_mean"][sku]
            for sku in cfg["sku_names"]
        ) * max_days

        if best_reward <= worst_reward:
            return 0.5

        score = (s.cumulative_reward - worst_reward) / (best_reward - worst_reward)
        return float(max(0.0, min(1.0, score)))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_obs(
        self,
        demand: Dict[str, int],
        message: str = "",
    ) -> SupplyChainObservation:
        s = self._state
        return SupplyChainObservation(
            day=s.day,
            retailer_stock=dict(s.retailer_stock),
            pipeline={sku: sum(s.pipeline[sku]) for sku in s.sku_names},
            demand=demand,
            cumulative_reward=s.cumulative_reward,
            message=message,
            done=s.done,
            task=s.task,
        )
