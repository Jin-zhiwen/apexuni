#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


ROUTES = ("lightglue", "lightglue_no_box", "dino_direct")
COUNTERS = ("triggers", "geometry_passes", "geometry_rejects", "approaches", "stops")


def main():
    parser = argparse.ArgumentParser(description="Summarize INSiNav route_metrics.jsonl")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    totals = {route: defaultdict(int) for route in ROUTES}
    episodes = 0
    successes = 0
    stop_successes = defaultdict(int)
    stop_episodes = defaultdict(int)

    with args.path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            episodes += 1
            successes += int(record["success"])
            for route in ROUTES:
                stats = record["routes"][route]
                for counter in COUNTERS:
                    totals[route][counter] += int(stats[counter])
                totals[route]["episodes_triggered"] += int(stats["triggers"] > 0)
            stop_route = record.get("stop_route")
            if stop_route in ROUTES:
                stop_episodes[stop_route] += 1
                stop_successes[stop_route] += int(record["success"])

    print(f"episodes={episodes} success={successes} SR={successes / max(episodes, 1):.4f}")
    header = (
        "route", "episodes", "triggers", "pass", "reject", "approach", "stops", "stop_SR"
    )
    print(" ".join(f"{column:>18}" for column in header))
    for route in ROUTES:
        stop_sr = stop_successes[route] / max(stop_episodes[route], 1)
        row = (
            route,
            totals[route]["episodes_triggered"],
            totals[route]["triggers"],
            totals[route]["geometry_passes"],
            totals[route]["geometry_rejects"],
            totals[route]["approaches"],
            totals[route]["stops"],
            f"{stop_sr:.4f}",
        )
        print(" ".join(f"{str(value):>18}" for value in row))


if __name__ == "__main__":
    main()
