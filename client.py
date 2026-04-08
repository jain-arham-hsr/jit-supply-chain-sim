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


class JITSupplyChainEnv(EnvClient):
    """Remote client for the JIT Supply Chain Simulator server."""

    action_type = SupplyChainAction
    observation_type = SupplyChainObservation
    state_type = SupplyChainState

    def _parse_observation(self, data: Dict[str, Any]) -> SupplyChainObservation:
        return SupplyChainObservation(**data)

    def _parse_state(self, data: Dict[str, Any]) -> SupplyChainState:
        return SupplyChainState(**data)
