"""Run agent_eval under different AGENT_TEMPERATURE / CONFIDENCE_THRESHOLD
values and compare, so production .env defaults are chosen with data.

Each configuration runs in its own subprocess: a fresh import re-reads the
env values (RAGAgent binds the threshold default at import time), and each
run's results are saved as agent_eval_<knob>_<value>.json.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

KNOB_ENV = {
    "temperature": "AGENT_TEMPERATURE",
    "confidence_threshold": "CONFIDENCE_THRESHOLD",
}

DEFAULT_VALUES = {
    "temperature": ["0.0", "0.2"],
    "confidence_threshold": ["0.5", "0.65", "0.8"],
}

AGENT_EVAL_JSON = RESULTS_DIR / "agent_eval.json"


def run_one(knob, value, results_dir=None):
    results_dir = results_dir or RESULTS_DIR
    env = {**os.environ, KNOB_ENV[knob]: str(value)}
    subprocess.run(
        [sys.executable, "-m", "evaluation.agent_eval"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    labeled = results_dir / f"agent_eval_{knob}_{value}.json"
    labeled.write_text(AGENT_EVAL_JSON.read_text())
    return json.loads(labeled.read_text())


def summarize(results):
    agentic = results["agentic_rag"]
    quality = agentic["answer_quality"]
    return {
        "mean_score": quality["mean_score"],
        "num_evaluated": quality["num_evaluated"],
        "retrieval_hit_rate": agentic["retrieval"]["hit_rate"],
        "avg_searches_per_query": agentic["avg_searches_per_query"],
        "latency_per_query": agentic["latency_per_query"],
        "total_time_seconds": agentic["total_time_seconds"],
    }


def compare(knob, values, results_dir=None):
    sweep = {}
    for value in values:
        print(f"\n=== {knob}={value} ===")
        sweep[str(value)] = summarize(run_one(knob, value, results_dir))
    return sweep


def print_table(knob, sweep):
    print("\n" + "=" * 72)
    print(f"CONFIG SWEEP: {knob}")
    print("=" * 72)
    print(f"{'value':<12} {'judge':>6} {'hit%':>7} {'searches':>9} {'latency':>9}")
    for value, metrics in sweep.items():
        print(
            f"{value:<12} {metrics['mean_score']:>6.2f} "
            f"{metrics['retrieval_hit_rate']:>7.1%} "
            f"{metrics['avg_searches_per_query']:>9.2f} "
            f"{metrics['latency_per_query']:>9.2f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knob",
        choices=sorted(KNOB_ENV),
        default="temperature",
        help="which env knob to sweep",
    )
    parser.add_argument(
        "--values",
        default=None,
        help="comma-separated values (default: %(default)s sweep set)",
    )
    args = parser.parse_args(argv)
    values = args.values.split(",") if args.values else DEFAULT_VALUES[args.knob]

    sweep = compare(args.knob, values)
    print_table(args.knob, sweep)
    sweep_path = RESULTS_DIR / f"config_sweep_{args.knob}.json"
    sweep_path.write_text(json.dumps(sweep, indent=2))
    print(f"\nSweep saved to {sweep_path}")
    return sweep


if __name__ == "__main__":
    main()
