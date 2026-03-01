"""Run the Code Forge self-correction pipeline.

Usage:
    python scripts/run_forge.py [-n ROUNDS] [--max-heal N]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import weave
from src.config import WEAVE_PROJECT
from pipeline.forge import run_forge


async def main():
    parser = argparse.ArgumentParser(description="Run the Code Forge")
    parser.add_argument("-n", "--rounds", type=int, default=3, help="Number of forge rounds")
    parser.add_argument("--max-heal", type=int, default=3, help="Max heal attempts per task")
    args = parser.parse_args()

    weave.init(WEAVE_PROJECT)
    results = await run_forge(n_rounds=args.rounds, max_heal_attempts=args.max_heal)
    return results


if __name__ == "__main__":
    asyncio.run(main())
