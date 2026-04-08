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
from typing import List, Optional

from openai import OpenAI

from client import JITSupplyChainEnv
from models import SupplyChainAction

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "registry.hf.space/jain-arham-hsr/jit-supply-chain-sim:latest")

if not API_KEY:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "jit_supply_chain_sim"
TASKS = ["easy_reorder", "medium_stochastic", "hard_multi_sku"]
MAX_STEPS = 100
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
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
    for each SKU. Issue replenishment orders to:
      - Prevent stockouts (very costly: -2.0 per unit short)
      - Avoid excess holding costs (-0.05 per unit held)
      - Account for lead times before orders arrive

    Respond ONLY with a JSON object mapping SKU names to integer order quantities.
    Example: {"sku_0": 15}
    No explanation, no markdown — only the JSON object.
""").strip()


def get_model_action(client: OpenAI, obs: dict, step: int) -> dict:
    user_msg = textwrap.dedent(f"""
        Day {obs.get('day', '?')} | Task: {obs.get('task', '?')}
        Retailer stock:     {json.dumps(obs.get('retailer_stock', {}))}
        Pipeline (transit): {json.dumps(obs.get('pipeline', {}))}
        Last demand:        {json.dumps(obs.get('demand', {}))}
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
        return {k: max(0, int(v)) for k, v in orders.items()}
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {}


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task: str) -> None:
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    steps_taken = 0
    rewards: List[float] = []
    score = 0.0
    success = False

    env = await JITSupplyChainEnv.from_docker_image(LOCAL_IMAGE_NAME)

    try:
        result = await env.reset(task=task)
        obs = result.observation.model_dump()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            orders = get_model_action(client, obs, step)
            action = SupplyChainAction(orders=orders)
            action_str = json.dumps(orders)

            result = await env.step(action)
            obs = result.observation.model_dump()

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        total = sum(rewards)
        max_possible = MAX_STEPS * 10.0
        score = float(max(0.0, min(1.0, total / max_possible)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        success = False

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())
