"""
JIT Supply Chain Simulator — Pydantic data models.

Action   : replenishment order(s) for each SKU
Observation : current inventory levels, pipeline orders, demand history, costs
State    : full internal state for serialisation / inspection
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SupplyChainAction(BaseModel):
    """
    Order quantities to issue to the supplier for each SKU.

    Keys are SKU identifiers (e.g. "sku_0", "sku_1", "sku_2").
    Values are non-negative integer order quantities.
    """

    orders: Dict[str, int] = Field(
        default_factory=dict,
        description="SKU → order quantity mapping (non-negative integers).",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SupplyChainObservation(BaseModel):
    """
    What the agent sees at the start of each step.
    """

    # Current step / day
    day: int = Field(..., description="Current simulation day (0-indexed).")

    # Inventory at the retailer for each SKU
    retailer_stock: Dict[str, int] = Field(
        ..., description="Units on-hand at the retailer."
    )

    # Units in the pipeline (ordered but not yet delivered) per SKU
    pipeline: Dict[str, int] = Field(
        ..., description="Units in transit to the retailer."
    )

    # Demand realised this step for each SKU
    demand: Dict[str, int] = Field(
        ..., description="Demand units fulfilled (or attempted) this step."
    )

    # Cumulative reward so far
    cumulative_reward: float = Field(
        ..., description="Total reward accumulated since episode reset."
    )

    # Textual description of what happened this step (for LLM agents)
    message: str = Field(
        default="", description="Human-readable summary of this step."
    )

    # Whether the episode is over
    done: bool = Field(default=False)

    # Task identifier
    task: str = Field(default="", description="Current task name.")


# ---------------------------------------------------------------------------
# State (full internal state)
# ---------------------------------------------------------------------------

class SupplyChainState(BaseModel):
    """
    Full serialisable snapshot of environment state (for debugging / logging).
    """

    task: str
    day: int
    max_days: int
    sku_names: List[str]
    retailer_stock: Dict[str, int]
    warehouse_stock: Dict[str, int]
    pipeline: Dict[str, List[int]]          # per SKU: list of pending deliveries
    cumulative_reward: float
    cumulative_holding_cost: float
    cumulative_stockout_cost: float
    done: bool
    episode_id: Optional[str] = None
    step_count: int = 0
