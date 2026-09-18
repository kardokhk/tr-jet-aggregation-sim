"""Configuration loading, grid-cell enumeration and per-cell seeding."""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np
import yaml

EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6")


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k, v in cfg["parameters"].items() if "source" not in v]
    if missing:
        raise ValueError(f"parameters without 'source': {missing}")
    return cfg


def base_params(cfg: dict) -> dict:
    return {k: copy.deepcopy(v["value"]) for k, v in cfg["parameters"].items()}


def _levels(cfg: dict, entry) -> tuple[str, list[dict]]:
    """Return (name, list of override dicts) for one `vary` entry."""
    if isinstance(entry, str):
        grid = cfg["parameters"][entry].get("grid")
        if grid is None:
            raise ValueError(f"parameter {entry} has no grid")
        return entry, [{entry: g} for g in grid]
    name = entry["name"]
    levels = []
    for lv in entry["levels"]:
        levels.append(dict(lv) if isinstance(lv, dict) else {name: lv})
    return name, levels


def experiment_cells(cfg: dict, exp: str) -> list[dict]:
    """All grid cells of an experiment as override dicts, in a fixed order."""
    ex = cfg["experiments"][exp]
    axes = [_levels(cfg, e) for e in ex.get("vary", [])]
    cells = []
    for combo in itertools.product(*[lv for _, lv in axes]):
        ov = {}
        for d in combo:
            ov.update(d)
        cells.append(ov)
    return cells


def cell_params(cfg: dict, exp: str, cell: int, overrides: dict | None = None) -> dict:
    p = base_params(cfg)
    p.update(copy.deepcopy(cfg["experiments"][exp].get("fixed", {}) or {}))
    p.update(experiment_cells(cfg, exp)[cell])
    if overrides:
        p.update(overrides)
    return p


def cell_seed(cfg: dict, exp: str, cell: int) -> np.random.SeedSequence:
    """Child of the master SeedSequence, keyed by (experiment, cell).

    Equivalent to SeedSequence(master).spawn(...) with a two-level key, and
    independent of how cells are chunked across CLI invocations.
    """
    e = EXPERIMENTS.index(exp) + 1
    return np.random.SeedSequence(entropy=cfg["meta"]["master_seed"], spawn_key=(e, cell))


def parse_cells(spec: str | None, n_cells: int) -> range:
    if spec is None or spec == "all":
        return range(n_cells)
    if ":" in spec:
        a, b = spec.split(":")
        a = int(a) if a else 0
        b = min(int(b), n_cells) if b else n_cells
        return range(a, b)
    i = int(spec)
    return range(i, i + 1)
