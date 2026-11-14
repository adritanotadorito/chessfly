"""Stockfish-labelled warm start and a small REINFORCE continuation."""
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np

from .env import ChessEnv
from .model import FlyPolicy


def generate(path: str, stockfish: str, games: int = 16, seed: int = 12,
             teacher_nodes: int = 5000):
    """Split by independently generated game, never by position from one game."""
    rng = np.random.default_rng(seed)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": 1, "Hash": 16})
    version = engine.id
    counts = {"train": 0, "heldout": 0}
    try:
        with open(path, "w") as f:
            f.write(json.dumps({"metadata": {"engine": version, "nodes": teacher_nodes,
                                           "seed": seed, "games": games}}) + "\n")
            for game in range(games):
                board = chess.Board()
                for _ in range(int(rng.integers(5, 13))):
                    if board.is_game_over():
                        break
                    board.push(list(board.legal_moves)[int(rng.integers(board.legal_moves.count()))])
                split = "heldout" if game % 5 == 0 else "train"
                for _ in range(4):
                    if board.is_game_over():
                        break
                    fen = board.fen()
                    move = engine.play(board, chess.engine.Limit(nodes=teacher_nodes)).move
                    f.write(json.dumps({"game": game, "split": split, "fen": fen,
                                        "teacher_move": move.uci()}) + "\n")
                    counts[split] += 1
                    board.push(move)
                    if board.is_game_over():
                        break
                    board.push(list(board.legal_moves)[int(rng.integers(board.legal_moves.count()))])
    finally:
        engine.quit()
    return {"engine": version, "examples": counts}


def read_examples(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    meta = rows[0].get("metadata")
    if not meta:
        raise ValueError("Dataset missing engine provenance")
    return meta, rows[1:]


def accuracy(model, examples):
    if not examples:
        return None
    return sum(model.choose(chess.Board(x["fen"])).uci() == x["teacher_move"]
               for x in examples) / len(examples)


def train_supervised(graph, dataset, output, epochs=4, seed=7, shuffled=False, lr=0.002):
    meta, examples = read_examples(dataset)
    train = [x for x in examples if x["split"] == "train"]
    held = [x for x in examples if x["split"] == "heldout"]
    if not train or not held:
        raise ValueError("Need both train and heldout positions")
    model = FlyPolicy(graph, seed=seed, shuffled=shuffled)
    before = accuracy(model, held)
    rng = np.random.default_rng(seed)
    logs = []
    for epoch in range(epochs):
        losses = []
        for i in rng.permutation(len(train)):
            row = train[int(i)]
            board = chess.Board(row["fen"])
            loss, grad = model.gradients(board, chess.Move.from_uci(row["teacher_move"]))
            model.update(grad, lr=lr)
            losses.append(loss)
        logs.append({"epoch": epoch+1, "train_loss": round(float(np.mean(losses)), 4),
                     "heldout_top1": accuracy(model, held)})
    model.save(output)
    return {"graph_authentic": model.graph_meta["authentic_flywire"],
            "shuffled": shuffled, "initial_heldout_top1": before,
            "history": logs, "dataset": meta, "checkpoint": output}


def train_reinforce(graph, checkpoint, output, episodes=30, seed=7,
                    opponent="random", stockfish=None, lr=0.0002):
    """Small policy-gradient experiment; not a claim of strong chess performance."""
    model = FlyPolicy(graph, seed=seed)
    model.load_weights(checkpoint)
    baseline = 0.0
    logs = []
    for episode in range(episodes):
        env = ChessEnv(opponent=opponent, stockfish=stockfish, seed=seed+episode,
                       max_agent_moves=6)
        trajectory = []
        try:
            while not env.done:
                board = env.board.copy(stack=True)
                move = model.choose(board, sample=True)
                info = env.play(move.uci())
                trajectory.append((board, move, info["training_reward"]))
            running = 0.0
            returns = []
            for _, _, reward in reversed(trajectory):
                running = reward + 0.99*running
                returns.append(running)
            returns.reverse()
            for (board, move, _), ret in zip(trajectory, returns):
                _, grads = model.gradients(board, move, scale=np.clip(ret-baseline, -2, 2))
                model.update(grads, lr=lr)
            baseline = 0.95*baseline + 0.05*(returns[0] if returns else 0)
            logs.append({"episode": episode+1, "steps": len(trajectory),
                         "return": round(returns[0], 4) if returns else 0,
                         "actual_result": info["actual_result"]})
        finally:
            env.close()
    model.save(output)
    return {"episodes": logs, "checkpoint": output,
            "warning": "Evaluate on held-out scenarios; training returns alone prove no improvement"}
