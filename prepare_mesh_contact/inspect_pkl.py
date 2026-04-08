#!/usr/bin/env python3
"""Inspect a pickle file and summarize its structure."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a .pkl clip structure")
    parser.add_argument("path", type=str, help="Path to the pickle file")
    parser.add_argument("--max-items", type=int, default=12, help="Max items to print per mapping/list")
    return parser.parse_args()


def summarize(value: Any, max_items: int, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        print(f"{prefix}dict[{len(value)}]")
        for idx, (key, item) in enumerate(sorted(value.items(), key=lambda kv: str(kv[0]))):
            if idx >= max_items:
                print(f"{prefix}  ... (+{len(value) - max_items} more keys)")
                break
            print(f"{prefix}  {key}:")
            summarize(item, max_items=max_items, indent=indent + 4)
        return

    if isinstance(value, (list, tuple)):
        print(f"{prefix}{type(value).__name__}[{len(value)}]")
        for idx, item in enumerate(value[:max_items]):
            print(f"{prefix}  [{idx}]:")
            summarize(item, max_items=max_items, indent=indent + 4)
        if len(value) > max_items:
            print(f"{prefix}  ... (+{len(value) - max_items} more items)")
        return

    if isinstance(value, np.ndarray):
        shape = tuple(int(x) for x in value.shape)
        print(f"{prefix}ndarray shape={shape} dtype={value.dtype}")
        return

    print(f"{prefix}{type(value).__name__}: {repr(value)[:160]}")


def main() -> None:
    args = parse_args()
    path = args.path
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        obj = pickle.load(f)

    print(f"Path: {path}")
    summarize(obj, max_items=max(1, args.max_items))


if __name__ == "__main__":
    main()
