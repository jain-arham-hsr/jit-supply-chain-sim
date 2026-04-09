"""
JIT Supply Chain Simulator — HTTP client.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from .models import SupplyChainAction, SupplyChainObservation, SupplyChainState
except ImportError:
    from models import SupplyChainAction, SupplyChainObservation, SupplyChainState

from openenv.core import EnvClient
from openenv.core.client_types import StepResult


class JITSupplyChainEnv(EnvClient[SupplyChainAction, SupplyChainObservation, SupplyChainState]):
    """Remote client for the JIT Supply Chain Simulator server."""

    def _step_payload(self, action: SupplyChainAction) -> dict:
        return {"orders": action.orders}

    def _parse_result(self, payload: dict) -> StepResult[SupplyChainObservation]:
        obs_data = payload.get("observation", {})
        obs = SupplyChainObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SupplyChainState:
        return SupplyChainState(**payload)
