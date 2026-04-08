"""
JIT Supply Chain Simulator — FastAPI server application.

Entry point: uvicorn server.app:app
"""

from __future__ import annotations
import uvicorn

try:
    from ..models import SupplyChainAction, SupplyChainObservation
    from .supply_chain_environment import SupplyChainEnvironment
except ImportError:
    from models import SupplyChainAction, SupplyChainObservation
    from server.supply_chain_environment import SupplyChainEnvironment

from openenv.core import create_app

app = create_app(
    SupplyChainEnvironment,
    SupplyChainAction,
    SupplyChainObservation,
    env_name="jit_supply_chain_sim",
)

def main():
    """
    The entry point for the 'server' command.
    """
    # We point uvicorn to this specific file and the 'app' variable defined above
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

# Fix 2: Add the execution guard
if __name__ == "__main__":
    main()
