#!/usr/bin/env python3
"""
Launcher for parallel synthetic Bhojpuri generation.
Usage:
  python3 run_synthetic_generation.py --worker-id a --generations-per-cycle 50
  # Or run in background:
  nohup python3 run_synthetic_generation.py --worker-id a > /tmp/synthetic_a.log 2>&1 &
"""
import argparse
import sys
from pathlib import Path

from .synthetic_bhojpuri_generator import SyntheticBhojpuriGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Bhojpuri text generator launcher (single worker)"
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        choices=['a', 'b', 'c'],
        default='a',
        help="Worker ID (a/b/c) - determines topic focus"
    )
    parser.add_argument(
        "--generations-per-cycle",
        type=int,
        default=50,
        help="Number of generations per cycle"
    )
    parser.add_argument(
        "--cycle-delay",
        type=int,
        default=30,
        help="Delay between cycles (seconds)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: bhojpuri/data)"
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=500_000_000,
        help="Target token count"
    )

    args = parser.parse_args()

    generator = SyntheticBhojpuriGenerator(
        worker_id=args.worker_id,
        data_dir=args.data_dir,
        generations_per_cycle=args.generations_per_cycle,
        cycle_delay_seconds=args.cycle_delay,
        target_tokens=args.target_tokens,
    )

    try:
        generator.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
