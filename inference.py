"""
JIT Supply Chain Simulator — Baseline Inference Script

Environment variables:
    HF_TOKEN          Your Hugging Face API key (mandatory, no default)
    API_BASE_URL      LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    LOCAL_IMAGE_NAME  Docker image name for the environment container

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from client import JITSupplyChainEnv
from models import SupplyChainAction

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME: str = os.getenv(
    "LOCAL_IMAGE_NAME",
    "registry.hf.space/jain-arham-hsr-jit-supply-chain-sim:latest",
)

if not API_KEY:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "jit_supply_chain_sim"
TASKS = ["easy_reorder", "medium_stochastic", "hard_multi_sku"]
MAX_STEPS = 100
MAX_CONSECUTIVE_LLM_FAILURES = 3   # stop episode if LLM fails this many times in a row
HISTORY_LENGTH = 5                  # how many past steps to show the LLM
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert supply-chain manager running a Just-In-Time replenishment system.
    Each day you observe retailer stock, in-transit pipeline inventory, and recent demand
    for each SKU. You also see a history of recent steps to help you adapt.

    Your goals:
      - Prevent stockouts (very costly: -2.0 per unit short)
      - Avoid excess holding costs (-0.05 per unit held per day)
      - Account for lead times: orders take several days to arrive

    Strategy tips:
      - Order enough to cover expected demand during the lead time
      - If stock is low and pipeline is empty, order more urgently
      - If stock is high, order less or nothing

    Respond ONLY with a JSON object mapping SKU names to integer order quantities.
    Example: {"sku_0": 15}
    No explanation, no markdown — only the JSON object.
""").strip()


def get_model_action(
    client: OpenAI,
    obs: dict,
    history: List[str],
) -> tuple[dict, bool]:
    """
    Returns (orders_dict, llm_succeeded).
    """
    history_block = "\n".join(history[-HISTORY_LENGTH:]) if history else "No history yet."

    user_msg = textwrap.dedent(f"""
        === Recent History (last {HISTORY_LENGTH} steps) ===
        {history_block}

        === Current State ===
        Day {obs.get('day', '?')} | Task: {obs.get('task', '?')}
        Retailer stock:     {json.dumps(obs.get('retailer_stock', {}))}
        Pipeline (transit): {json.dumps(obs.get('pipeline', {}))}
        Last demand:        {json.dumps(obs.get('demand', {}))}
        Last reward:        {obs.get('reward', 0):.2f}
        Cumulative reward:  {obs.get('cumulative_reward', 0):.2f}
        Message: {obs.get('message', '')}

        Decide replenishment orders (JSON only):
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        orders = json.loads(raw.strip())
        return {k: max(0, int(v)) for k, v in orders.items()}, True
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {}, False


def compute_score(rewards: List[float], obs: dict) -> float:
    """
    Compute normalised score [0, 1].
    Uses cumulative_reward from the observation relative to task best/worst case.
    Falls back to a reward-sum heuristic if needed.
    """
    cumulative = obs.get("cumulative_reward", sum(rewards))
    # Rough normalisation: assume best case ~9.5/step (fulfill 10 - hold 0.5)
    # and worst case ~-20.5/step (stockout 10*2 + hold 0.5)
    steps = max(len(rewards), 1)
    best = 9.5 * steps
    worst = -20.5 * steps
    if best <= worst:
        return 0.5
    score = (cumulative - worst) / (best - worst)
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, env: JITSupplyChainEnv, task: str) -> None:
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    steps_taken = 0
    rewards: List[float] = []
    history: List[str] = []
    score = 0.0
    success = False
    consecutive_failures = 0
    last_obs: dict = {}

    try:
        result = await env.reset(task=task)
        obs = result.observation.model_dump()
        last_obs = obs

        for step in range(1, MAX_STEPS + 1):

            orders, llm_ok = get_model_action(client, obs, history)

            if not llm_ok:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
                    print(
                        f"[DEBUG] {MAX_CONSECUTIVE_LLM_FAILURES} consecutive LLM failures — stopping episode.",
                        flush=True,
                    )
                    break
            else:
                consecutive_failures = 0

            action = SupplyChainAction(orders=orders)
            action_str = json.dumps(orders)

            result = await env.step(action)
            obs = result.observation.model_dump()
            last_obs = obs

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step

            # Build history entry for next step
            history.append(
                f"Day {obs.get('day', '?')}: ordered {action_str} | "
                f"demand={json.dumps(obs.get('demand', {}))} | "
                f"stock={json.dumps(obs.get('retailer_stock', {}))} | "
                f"reward={reward:.2f}"
            )

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        score = compute_score(rewards, last_obs)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Spin up the container once and reuse across all tasks
    env = await JITSupplyChainEnv.from_docker_image(LOCAL_IMAGE_NAME)
    try:
        for task in TASKS:
            await run_task(client, env, task)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
