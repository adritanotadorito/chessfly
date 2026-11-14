"""Small NumPy policy. Every trainable recurrent edge belongs to the fixed graph.

The dense board adapter and move readout are engineered; they are not fly neurons.
"""
import json
from pathlib import Path

import chess
import numpy as np

from .board import DIM, encode, move_parts
from .graph import load_graph


class FlyPolicy:
    def __init__(self, graph_path: str, seed: int = 7, shuffled: bool = False, passes: int = 3):
        self.src, self.dst, self.weight, self.graph_meta, self.n = load_graph(graph_path, shuffled, seed)
        self.shuffled, self.passes = shuffled, passes
        rng = np.random.default_rng(seed)
        self.params = {
            "adapter": rng.normal(0, 0.02, (self.n, DIM)).astype(np.float32),
            "adapter_bias": np.zeros(self.n, np.float32),
            "gain": np.ones(len(self.src), np.float32),
            "from": rng.normal(0, 0.05, (64, self.n)).astype(np.float32),
            "to": rng.normal(0, 0.05, (64, self.n)).astype(np.float32),
            "promo": rng.normal(0, 0.05, (5, self.n)).astype(np.float32),
        }
        self.rng = rng
        self.steps = 0
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, board: chess.Board, my_clock: float = None, opp_clock: float = None):
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves in terminal position")
        features = encode(board, my_clock=my_clock, opp_clock=opp_clock)
        x = np.tanh(self.params["adapter"] @ features + self.params["adapter_bias"])
        states = [x]
        for _ in range(self.passes):
            messages = self.weight * self.params["gain"] * x[self.src]
            x = np.tanh(0.5*x + np.bincount(self.dst, weights=messages, minlength=self.n))
            states.append(x)
        parts = np.array([move_parts(m) for m in legal], dtype=np.int64)
        vectors = (self.params["from"][parts[:, 0]] +
                   self.params["to"][parts[:, 1]] + self.params["promo"][parts[:, 2]])
        logits = vectors @ x
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return legal, logits, probs, (features, states, parts, vectors)

    def choose(self, board: chess.Board, sample: bool = False, my_clock: float = None, opp_clock: float = None):
        legal, _, probs, _ = self.forward(board, my_clock=my_clock, opp_clock=opp_clock)
        idx = self.rng.choice(len(legal), p=probs) if sample else int(np.argmax(probs))
        return legal[idx]

    def gradients(self, board: chess.Board, target: chess.Move, scale: float = 1.0,
                  my_clock: float = None, opp_clock: float = None):
        legal, logits, probs, (features, states, parts, vectors) = self.forward(
            board, my_clock=my_clock, opp_clock=opp_clock)
        if target not in legal:
            raise ValueError(f"Teacher move {target.uci()} is illegal in {board.fen()}")
        idx = legal.index(target)
        loss = -float(np.log(max(probs[idx], 1e-12)))
        dlogits = probs.copy()
        dlogits[idx] -= 1.0
        dlogits *= scale
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dx = dlogits @ vectors
        for k, column in (("from", 0), ("to", 1), ("promo", 2)):
            np.add.at(grads[k], parts[:, column], dlogits[:, None] * states[-1][None, :])
        for t in range(self.passes - 1, -1, -1):
            dpre = dx * (1 - states[t+1]**2)
            np.add.at(grads["gain"], np.arange(len(self.src)),
                      dpre[self.dst] * self.weight * states[t][self.src])
            dx = 0.5*dpre + np.bincount(self.src,
                    weights=dpre[self.dst] * self.weight * self.params["gain"], minlength=self.n)
        dpre = dx * (1 - states[0]**2)
        grads["adapter"] = np.outer(dpre, features).astype(np.float32)
        grads["adapter_bias"] = dpre.astype(np.float32)
        return loss, grads

    def update(self, grads, lr: float = 0.002):
        self.steps += 1
        norm = np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        clip = min(1.0, 5.0 / max(norm, 1e-12))
        for k, g in grads.items():
            g = g * clip
            self.m[k] = 0.9*self.m[k] + 0.1*g
            self.v[k] = 0.999*self.v[k] + 0.001*(g*g)
            mhat = self.m[k] / (1 - 0.9**self.steps)
            vhat = self.v[k] / (1 - 0.999**self.steps)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        self.params["gain"] = np.clip(self.params["gain"], -2, 2)

    def save(self, path: str):
        header = {"graph_meta": self.graph_meta, "shuffled": self.shuffled,
                  "passes": self.passes, "n": self.n, "edges": len(self.src)}
        np.savez_compressed(path, header=json.dumps(header),
                            **{f"p_{k}": v for k, v in self.params.items()})

    def load_weights(self, path: str):
        data = np.load(path, allow_pickle=False)
        header = json.loads(str(data["header"]))
        if (header["graph_meta"] != self.graph_meta or header["shuffled"] != self.shuffled
                or header["passes"] != self.passes or header["edges"] != len(self.src)):
            raise ValueError("Checkpoint graph/config does not match")
        for k in self.params:
            param_data = data[f"p_{k}"]
            # Support backward compatibility and safe migration from 783-dim inputs
            if k == "adapter" and param_data.shape == (self.n, 783) and self.params["adapter"].shape == (self.n, DIM):
                padded = np.pad(param_data, ((0, 0), (0, DIM - 783)), mode="constant", constant_values=0.0)
                self.params["adapter"] = padded.astype(np.float32)
            elif param_data.shape != self.params[k].shape:
                raise ValueError(f"Shape mismatch for {k}: file has {param_data.shape}, expected {self.params[k].shape}")
            else:
                self.params[k] = param_data.copy()


def migrate_checkpoint(source_path: str, dest_path: str):
    """Safely migrate legacy 783-dim adapter checkpoint to 785-dim clock representation without touching the source."""
    data = np.load(source_path, allow_pickle=False)
    header = json.loads(str(data["header"]))
    saved_params = {}
    for k in ["adapter", "adapter_bias", "gain", "from", "to", "promo"]:
        v = data[f"p_{k}"]
        if k == "adapter" and v.shape[1] == 783:
            v = np.pad(v, ((0, 0), (0, 2)), mode="constant", constant_values=0.0).astype(np.float32)
        saved_params[f"p_{k}"] = v
    header["dim"] = DIM
    header["migrated_clock_inputs"] = True
    np.savez_compressed(dest_path, header=json.dumps(header), **saved_params)


def summarize(board: chess.Board, model: FlyPolicy, top: int = 5, my_clock: float = None, opp_clock: float = None):
    legal, logits, probs, _ = model.forward(board, my_clock=my_clock, opp_clock=opp_clock)
    indices = np.argsort(-probs)[:top]
    return [(legal[i].uci(), round(float(probs[i]), 4)) for i in indices]
