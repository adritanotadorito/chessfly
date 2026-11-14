import argparse
import json
from pathlib import Path

import chess

from .env import ChessEnv
from .graph import build_graph, make_fixture
from .model import FlyPolicy, summarize
from .match import BulletMatch
from .reward import compute_game_rewards
from .train import (generate, train_supervised, train_reinforce, read_examples,
                    accuracy, evaluate_heldout, evaluate_move_quality,
                    train_rl_bullet, evaluate_bullet_matches)


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
    train.add_argument("--lr", type=float, default=0.002)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--eval-stockfish")
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
    test.add_argument("--stockfish")
    bullet = sub.add_parser("bullet", help="Play a 1+0 bullet match against Stockfish")
    bullet.add_argument("--graph", required=True)
    bullet.add_argument("--checkpoint", required=True)
    bullet.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    bullet.add_argument("--color", choices=("white", "black"), default="white")
    bullet.add_argument("--difficulty", choices=("weak", "medium", "strong"), default="weak")
    bullet.add_argument("--shuffled", action="store_true")
    bullet.add_argument("--pgn-out")
    trl = sub.add_parser("train-rl", help="Run real online policy-gradient training from bullet games")
    trl.add_argument("--graph", required=True)
    trl.add_argument("--checkpoint", required=True)
    trl.add_argument("--out", required=True)
    trl.add_argument("--matches", type=int, default=6)
    trl.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    trl.add_argument("--difficulty", choices=("weak", "medium", "strong"), default="weak")
    trl.add_argument("--lr", type=float, default=0.0005)
    trl.add_argument("--seed", type=int, default=7)
    trl.add_argument("--shuffled", action="store_true")
    ebullet = sub.add_parser("eval-bullet", help="Evaluate bullet match performance across multiple games")
    ebullet.add_argument("--graph", required=True)
    ebullet.add_argument("--checkpoint", required=True)
    ebullet.add_argument("--games", type=int, default=4)
    ebullet.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    ebullet.add_argument("--difficulty", choices=("weak", "medium", "strong"), default="weak")
    ebullet.add_argument("--shuffled", action="store_true")
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
                                          args.epochs, seed=args.seed,
                                          shuffled=args.shuffled, lr=args.lr,
                                          eval_engine=args.eval_stockfish), indent=2))
    elif args.cmd == "reinforce":
        print(json.dumps(train_reinforce(args.graph, args.checkpoint, args.out,
                         args.episodes, opponent=args.opponent, stockfish=args.stockfish), indent=2))
    elif args.cmd == "heldout":
        model = FlyPolicy(args.graph, shuffled=args.shuffled)
        model.load_weights(args.checkpoint)
        _, rows = read_examples(args.dataset)
        held = [r for r in rows if r["split"] == "heldout"]
        results = evaluate_heldout(model, held)
        results["authentic_flywire"] = model.graph_meta["authentic_flywire"]
        results["heldout_top1"] = results["top1"]
        if args.stockfish:
            quality = evaluate_move_quality(model, held, args.stockfish)
            results["move_quality"] = quality
        print(json.dumps(results, indent=2))
    elif args.cmd == "bullet":
        model = FlyPolicy(args.graph, shuffled=args.shuffled)
        model.load_weights(args.checkpoint)
        fly_c = chess.WHITE if args.color == "white" else chess.BLACK
        match = BulletMatch(fly_model=model, stockfish_path=args.stockfish,
                            fly_color=fly_c, bot_difficulty=args.difficulty)
        res = match.run()
        if args.pgn_out:
            Path(args.pgn_out).write_text(res["pgn"])
        # Post-game transparent rewards
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        try:
            rewards = compute_game_rewards(res, engine=engine)
            res["reward_analysis"] = rewards
        finally:
            engine.quit()
        summary = {k: v for k, v in res.items() if k not in ("moves", "pgn")}
        print(json.dumps(summary, indent=2))
        print("\nPGN:")
        print(res["pgn"])
    elif args.cmd == "train-rl":
        res = train_rl_bullet(args.graph, args.checkpoint, args.out,
                              matches=args.matches, seed=args.seed,
                              shuffled=args.shuffled, stockfish_path=args.stockfish,
                              bot_difficulty=args.difficulty, lr=args.lr)
        print(json.dumps(res, indent=2))
    elif args.cmd == "eval-bullet":
        model = FlyPolicy(args.graph, shuffled=args.shuffled)
        model.load_weights(args.checkpoint)
        res = evaluate_bullet_matches(model, stockfish_path=args.stockfish,
                                      num_games=args.games, bot_difficulty=args.difficulty)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
