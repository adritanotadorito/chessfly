"""Extract an explicitly fly-derived, connected subgraph from FlyWire connections.csv.

This is a computational topology experiment, not a biological sensory circuit.
"""
import csv
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np


def build_graph(csv_path: str, out_path: str, nodes: int = 256, pool: int = 200000,
                confirmed_flywire: bool = False):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Supply the real train-your-fly connections.csv: {path}")
    if nodes < 8:
        raise ValueError("Use at least 8 neurons")
    digest = hashlib.sha256()
    best = []
    with path.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"pre_root_id", "post_root_id", "syn_count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected columns {sorted(required)}, got {reader.fieldnames}")
        for row in reader:
            try:
                a, b, w = int(row["pre_root_id"]), int(row["post_root_id"]), int(row["syn_count"])
            except (ValueError, TypeError):
                continue
            if a == b or w <= 0:
                continue
            edge = (w, a, b)
            if len(best) < pool:
                heapq.heappush(best, edge)
            elif edge > best[0]:
                heapq.heapreplace(best, edge)
    if not best:
        raise ValueError("No usable connections")
    edges = sorted(best, reverse=True)
    chosen = {edges[0][1], edges[0][2]}
    while len(chosen) < nodes:
        additions = []
        for w, a, b in edges:
            if (a in chosen) ^ (b in chosen):
                additions.append((w, b if a in chosen else a))
        if not additions:
            raise ValueError("Top-edge pool did not yield a connected subgraph; increase --pool")
        for _, candidate in sorted(additions, reverse=True):
            chosen.add(candidate)
            if len(chosen) == nodes:
                break
    ids = sorted(chosen)
    lookup = {root_id: i for i, root_id in enumerate(ids)}
    internal = [(lookup[a], lookup[b], w) for w, a, b in edges if a in chosen and b in chosen]
    if not internal:
        raise ValueError("Selected nodes have no connections")
    src, dst, weight = (np.array(v, dtype=np.int64) for v in zip(*internal))
    incoming = np.bincount(dst, weights=weight, minlength=nodes)
    weight = (weight / np.maximum(incoming[dst], 1)).astype(np.float32)
    metadata = {"source": str(path.name), "source_sha256": digest.hexdigest(),
                "selection": "highest-synapse connected expansion", "nodes": nodes,
                "edges": len(internal), "pool": pool,
                "authentic_flywire": confirmed_flywire,
                "provenance_note": "User confirmed upstream FlyWire CSV" if confirmed_flywire
                                   else "Unverified CSV schema; cannot claim fly provenance"}
    np.savez_compressed(out_path, src=src, dst=dst, weight=weight,
                        root_ids=np.array(ids, dtype=np.int64), meta=json.dumps(metadata))
    return metadata


def make_fixture(out_path: str):
    """Synthetic wiring for software checks ONLY. Never use for fly claims."""
    n = 16
    src = np.array([i for i in range(n) for _ in (0, 1)], dtype=np.int64)
    dst = np.array([(i + delta) % n for i in range(n) for delta in (1, 3)], dtype=np.int64)
    np.savez_compressed(out_path, src=src, dst=dst,
                        weight=np.full(2*n, 0.5, dtype=np.float32),
                        root_ids=np.arange(n), meta=json.dumps({"authentic_flywire": False,
                            "warning": "Synthetic fixture; no biological claim"}))


def load_graph(path: str, shuffled: bool = False, seed: int = 42):
    data = np.load(path, allow_pickle=False)
    src, dst = data["src"].copy(), data["dst"].copy()
    weight = data["weight"].copy()
    num_nodes = len(data["root_ids"])
    if shuffled:
        rng = np.random.default_rng(seed)
        dst = dst[rng.permutation(len(dst))]
        # Re-normalize incoming edge weights so sum of incoming weights per destination node
        # equals 1.0 (matching the biological graph scaling and preventing tanh saturation/vanishing gradients).
        incoming = np.bincount(dst, weights=weight, minlength=num_nodes)
        weight = (weight / np.maximum(incoming[dst], 1e-8)).astype(np.float32)
    return src, dst, weight, json.loads(str(data["meta"])), num_nodes
