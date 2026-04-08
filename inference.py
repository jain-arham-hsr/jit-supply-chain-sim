"""
JIT Supply Chain Simulator — Baseline Inference Script

Usage:
    export HF_TOKEN=<your_token>
    export API_BASE_URL=https://api.openai.com/v1   # optional
    export MODEL_NAME=gpt-4.1-mini                  # optional
    python inference.py

Output format (stdout):
    [START] task=<task_name> env=jit_supply_chain_sim model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, List

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "jit_supply_chain_sim"
TASKS = ["easy_reorder", "medium_stochastic", "hard_multi_sku"]
MAX_STEPS_PER_TASK = 100  # safety cap; env will signal done earlier


# ---------------------------------------------------------------------------
# Lightweight HTTP helpers (avoids importing openenv client-side)
# ---------------------------------------------------------------------------

def env_reset(task: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_close() -> None:
    try:
        requests.post(f"{ENV_BASE_URL}/reset", json={}, timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# LLM-based agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert supply-chain manager operating a Just-In-Time
replenishment system. Each day you observe retailer stock levels, in-transit
(pipeline) inventory, and the previous day's demand for each SKU. Your goal is to
issue replenishment orders that:
  - Prevent stockouts (costly)
  - Avoid excess holding costs
  - Account for lead times before orders arrive

Respond ONLY with a JSON object mapping SKU names to integer order quantities.
Example: {"sku_0": 15}
Do not include any explanation, markdown, or other text — only the JSON object."""


def agent_decide(observation: Dict[str, Any]) -> Dict[str, int]:
    """Ask the LLM to decide order quantities given the current observation."""
    user_msg = (
        f"Day {observation.get('day', '?')} | "
        f"Task: {observation.get('task', '?')}\n"
        f"Retailer stock: {json.dumps(observation.get('retailer_stock', {}))}\n"
        f"Pipeline (in transit): {json.dumps(observation.get('pipeline', {}))}\n"
        f"Last demand: {json.dumps(observation.get('demand', {}))}\n"
        f"Cumulative reward so far: {observation.get('cumulative_reward', 0):.2f}\n"
        f"Message: {observation.get('message', '')}\n\n"
        "Decide replenishment orders (JSON only):"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content.strip()

    # Parse JSON; fall back to zero orders on failure
    try:
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        orders = json.loads(raw)
        return {k: max(0, int(v)) for k, v in orders.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    step_num = 0
    rewards: List[float] = []
    success = False
    last_error: str | None = None

    try:
        obs = env_reset(task)
        done = obs.get("done", False)

        while not done and step_num < MAX_STEPS_PER_TASK:
            step_num += 1
            action_map: Dict[str, int] = {}
            action_str = "null"
            reward = 0.0
            error_str = "null"

            try:
                action_map = agent_decide(obs)
                action_str = json.dumps(action_map)

                result = env_step({"orders": action_map})
                obs = result.get("observation", result)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                last_error = result.get("error") or None
                error_str = last_error if last_error else "null"

            except Exception as exc:
                error_str = str(exc).replace("\n", " ")[:200]
                last_error = error_str
                done = True

            rewards.append(reward)
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={done_str} error={error_str}",
                flush=True,
            )

        success = done and (last_error is None)

    except Exception as exc:
        success = False
        last_error = str(exc).replace("\n", " ")[:200]
        if step_num == 0:
            rewards = []

    finally:
        env_close()

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={step_num} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tasks_to_run = sys.argv[1:] if len(sys.argv) > 1 else TASKS
    for task in tasks_to_run:
        run_task(task)
