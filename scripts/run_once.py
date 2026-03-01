"""Single evaluation run entry point."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import weave
from src.config import WEAVE_PROJECT
from src.generator import LuauCodeGenerator
from pipeline.evaluate import run_evaluation


async def main():
    weave.init(WEAVE_PROJECT)

    generator = LuauCodeGenerator()
    print("Running single evaluation...")
    print(f"  Model: {generator.model_name}")
    print(f"  Prompt length: {len(generator.system_prompt)} chars")
    print()

    results = await run_evaluation(generator)

    print("\n=== Results ===")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
