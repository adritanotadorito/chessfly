import argparse
import json
from pathlib import Path

import chess

from .env import ChessEnv
from .graph import build_graph, make_fixture
from .model import FlyPolicy, summarize
from .train import generate, train_supervised, train_reinforce, read_examples, accuracy


def main():
    p = argparse.ArgumentParser(prog="python -m flychess.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    fixture = sub.add_parser("fixture", help="Make synthetic graph for plumbing checks only")
    fixture.add_argument("out")
    graph = sub.add_parser("prepare-graph", help="Extract real FlyWire subgraph")
    graph.add_argument("connections")
    graph.add_argument("out")
    graph.add_argument("--nodes", type=int, default=256)
    graph.add_argument("--pool", type=int, default=200000)
    graph.add_argument("--confirm-flywire", action="store_true",
                       help="Only if this CSV is from the upstream FlyWire dataset")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--graph", required=True)
    smoke.add_argument("--checkpoint")
    smoke.add_argument("--shuffled", action="store_true")
    data = sub.add_parser("generate")
    data.add_argument("--stockfish", required=True)
    data.add_argument("--out", required=True)
    data.add_argument("--games", type=int, default=16)
    train = sub.add_parser("train")
    train.add_argument("--graph", required=True)
    train.add_argument("--dataset", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--shuffled", action="store_true")
    rl = sub.add_parser("reinforce")
    rl.add_argument("--graph", required=True)
    rl.add_argument("--checkpoint", required=True)
    rl.add_argument("--out", required=True)
    rl.add_argument("--episodes", type=int, default=30)
    rl.add_argument("--opponent", choices=("random", "stockfish"), default="random")
    rl.add_argument("--stockfish")
    test = sub.add_parser("heldout")
    test.add_argument("--graph", required=True)
    test.add_argument("--dataset", required=True)
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--shuffled", action="store_true")
    args = p.parse_args()
    if args.cmd == "fixture":
        make_fixture(args.out)
        print("Synthetic graph generated. This is NOT fly wiring.")
    elif args.cmd == "prepare-graph":
        print(json.dumps(build_graph(args.connections, args.out, args.nodes, args.pool,
                                     args.confirm_flywire), indent=2))
    elif args.cmd == "smoke":
        model = FlyPolicy(args.graph, shuffled=args.shuffled)
        if args.checkpoint:
            model.load_weights(args.checkpoint)
        print("Graph:", json.dumps(model.graph_meta))
        board = chess.Board()
        print("Opening move scores:", summarize(board, model))
        env = ChessEnv(seed=42, max_agent_moves=4)
        try:
            while not env.done:
                result = env.play(model.choose(env.board).uci())
            print("Finished short episode:", env.observe()["history"])
            print("Last transition:", {k: v for k, v in result.items() if k != "observation"})
        finally:
            env.close()
        full = ChessEnv(seed=42, max_agent_moves=200)
        try:
            while not full.done:
                full.play(model.choose(full.board).uci())
            print("Full game:", {"plies": len(full.history),
                                 "result": full.board.result(claim_draw=True),
                                 "terminal": full.board.is_game_over(claim_draw=True)})
        finally:
            full.close()
    elif args.cmd == "generate":
        print(json.dumps(generate(args.out, args.stockfish, args.games), indent=2))
    elif args.cmd == "train":
        print(json.dumps(train_supervised(args.graph, args.dataset, args.out,
                                          args.epochs, shuffled=args.shuffled), indent=2))
    elif args.cmd == "reinforce":
        print(json.dumps(train_reinforce(args.graph, args.checkpoint, args.out,
                         args.episodes, opponent=args.opponent, stockfish=args.stockfish), indent=2))
    elif args.cmd == "heldout":
        model = FlyPolicy(args.graph, shuffled=args.shuffled)
        model.load_weights(args.checkpoint)
        _, rows = read_examples(args.dataset)
        print(json.dumps({"heldout_top1": accuracy(model,
                          [r for r in rows if r["split"] == "heldout"]),
                          "authentic_flywire": model.graph_meta["authentic_flywire"]}, indent=2))


if __name__ == "__main__":
    main()
