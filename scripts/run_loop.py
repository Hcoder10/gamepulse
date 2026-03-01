"""Entry point: python scripts/run_loop.py -n 5"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import weave
from src.config import WEAVE_PROJECT
from pipeline.loop import run_loop


def main():
    parser = argparse.ArgumentParser(description="Run the Luau codegen self-improvement loop")
    parser.add_argument("-n", "--iterations", type=int, default=5, help="Number of iterations (default: 5)")
    args = parser.parse_args()

    weave.init(WEAVE_PROJECT)

    print(f"Weave project: {WEAVE_PROJECT}")
    print(f"Iterations: {args.iterations}")

    asyncio.run(run_loop(args.iterations))


if __name__ == "__main__":
    main()
