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
                    train_rl_bullet, evaluate_bullet_matches,
                    train_supervised_pilot, evaluate_frozen_benchmark,
                    evaluate_audited_chess_cases)
from .puzzle import (build_puzzle_pilot, evaluate_model_on_puzzles,
                     train_supervised_puzzles, train_puzzle_rl_episodes,
                     build_puzzle_10k_dataset, train_supervised_puzzle_10k,
                     compare_puzzle_evaluations)


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
    
    tpilot = sub.add_parser("train-pilot", help="Train supervised distillation pilot on Jan-Sep data with Oct validation")
    tpilot.add_argument("--graph", required=True)
    tpilot.add_argument("--train", default="data/training/train_2000.jsonl")
    tpilot.add_argument("--val", default="data/training/val_oct_200.jsonl")
    tpilot.add_argument("--out", required=True)
    tpilot.add_argument("--epochs", type=int, default=5)
    tpilot.add_argument("--seed", type=int, default=42)
    tpilot.add_argument("--shuffled", action="store_true")
    tpilot.add_argument("--lr", type=float, default=0.002)
    tpilot.add_argument("--value-weight", type=float, default=0.5)

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
    trl = sub.add_parser("train-rl", help="Run online Advantage Actor-Critic training from bullet games")
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

    ebench = sub.add_parser("eval-benchmark", help="Evaluate model on frozen November benchmark")
    ebench.add_argument("--graph", required=True)
    ebench.add_argument("--checkpoint")
    ebench.add_argument("--benchmark", default="data/benchmarks/november_benchmark_120.jsonl")
    ebench.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    ebench.add_argument("--shuffled", action="store_true")

    audit = sub.add_parser("audit-cases", help="Audit the 4 specific chess scenarios & repetition resistance")
    audit.add_argument("--graph")
    audit.add_argument("--checkpoint")
    audit.add_argument("--shuffled", action="store_true")

    # Puzzle pilot commands
    bpuzz = sub.add_parser("build-puzzles", help="Stream & build reproducible puzzle splits from lichess_db_puzzle.csv.zst")
    bpuzz.add_argument("--csv", default="data/lichess/lichess_db_puzzle.csv.zst")
    bpuzz.add_argument("--benchmark", default="data/benchmarks/november_benchmark_120.jsonl")
    bpuzz.add_argument("--out-dir", default="data/puzzles")
    bpuzz.add_argument("--train-per-bucket", type=int, default=40)
    bpuzz.add_argument("--val-per-bucket", type=int, default=10)
    bpuzz.add_argument("--test-per-bucket", type=int, default=10)

    epuzz = sub.add_parser("eval-puzzles", help="Evaluate model accuracy across puzzle themes")
    epuzz.add_argument("--graph", required=True)
    epuzz.add_argument("--checkpoint", required=True)
    epuzz.add_argument("--puzzles", default="data/puzzles/test_puzzles_frozen.jsonl")
    epuzz.add_argument("--shuffled", action="store_true")

    tpuzz = sub.add_parser("train-puzzles", help="Supervised training on puzzle pilot")
    tpuzz.add_argument("--graph", required=True)
    tpuzz.add_argument("--parent", required=True)
    tpuzz.add_argument("--train", default="data/puzzles/train_puzzles.jsonl")
    tpuzz.add_argument("--val", default="data/puzzles/val_puzzles.jsonl")
    tpuzz.add_argument("--out", required=True)
    tpuzz.add_argument("--epochs", type=int, default=5)
    tpuzz.add_argument("--lr", type=float, default=0.002)
    tpuzz.add_argument("--seed", type=int, default=42)
    tpuzz.add_argument("--shuffled", action="store_true")

    rlpuzz = sub.add_parser("train-puzzle-rl", help="Online RL puzzle episode training")
    rlpuzz.add_argument("--graph", required=True)
    rlpuzz.add_argument("--parent", required=True)
    rlpuzz.add_argument("--train", default="data/puzzles/train_puzzles.jsonl")
    rlpuzz.add_argument("--out", required=True)
    rlpuzz.add_argument("--episodes", type=int, default=50)
    rlpuzz.add_argument("--lr", type=float, default=0.001)
    rlpuzz.add_argument("--seed", type=int, default=42)
    rlpuzz.add_argument("--shuffled", action="store_true")

    # 10k Puzzle commands
    b10k = sub.add_parser("build-puzzles-10k", help="Stream & build 12,000 balanced puzzle splits from lichess_db_puzzle.csv.zst")
    b10k.add_argument("--csv", default="data/lichess/lichess_db_puzzle.csv.zst")
    b10k.add_argument("--november", default="data/benchmarks/november_benchmark_120.jsonl")
    b10k.add_argument("--october-val", default="data/training/val_oct_200.jsonl")
    b10k.add_argument("--out-dir", default="data/puzzles_10k")
    b10k.add_argument("--train-per-bucket", type=int, default=2000)
    b10k.add_argument("--val-per-bucket", type=int, default=200)
    b10k.add_argument("--test-per-bucket", type=int, default=200)
    b10k.add_argument("--seed", type=int, default=42)

    t10k = sub.add_parser("train-puzzles-10k", help="Supervised training on 10k puzzles interleaved with 2k chess positions")
    t10k.add_argument("--graph", required=True)
    t10k.add_argument("--parent", required=True)
    t10k.add_argument("--train-puzzles", default="data/puzzles_10k/train_puzzles_10k.jsonl")
    t10k.add_argument("--val-puzzles", default="data/puzzles_10k/val_puzzles_1000.jsonl")
    t10k.add_argument("--oct-val", default="data/training/val_oct_200.jsonl")
    t10k.add_argument("--train-chess", default="data/training/train_2000.jsonl")
    t10k.add_argument("--out", required=True)
    t10k.add_argument("--epochs", type=int, default=5)
    t10k.add_argument("--lr", type=float, default=0.001)
    t10k.add_argument("--seed", type=int, default=42)
    t10k.add_argument("--shuffled", action="store_true")

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
    elif args.cmd == "train-pilot":
        res = train_supervised_pilot(
            graph=args.graph,
            train_path=args.train,
            val_path=args.val,
            output=args.out,
            epochs=args.epochs,
            seed=args.seed,
            shuffled=args.shuffled,
            lr=args.lr,
            value_weight=args.value_weight
        )
        print(json.dumps(res, indent=2))
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
    elif args.cmd == "eval-benchmark":
        model = FlyPolicy(args.graph, shuffled=args.shuffled)
        if args.checkpoint:
            model.load_weights(args.checkpoint)
        res = evaluate_frozen_benchmark(model, benchmark_path=args.benchmark, stockfish_path=args.stockfish)
        print(json.dumps(res, indent=2))
    elif args.cmd == "audit-cases":
        model = None
        if args.graph and args.checkpoint:
            model = FlyPolicy(args.graph, shuffled=args.shuffled)
            model.load_weights(args.checkpoint)
        res = evaluate_audited_chess_cases(model)
        print(json.dumps(res, indent=2))
    elif args.cmd == "build-puzzles":
        manifest = build_puzzle_pilot(
            csv_zst_path=args.csv,
            benchmark_path=args.benchmark,
            output_dir=args.out_dir,
            target_train_per_bucket=args.train_per_bucket,
            target_val_per_bucket=args.val_per_bucket,
            target_test_per_bucket=args.test_per_bucket,
        )
        print(json.dumps(manifest, indent=2))
    elif args.cmd == "eval-puzzles":
        model = FlyPolicy(args.graph, shuffled=args.shuffled)
        model.load_weights(args.checkpoint)
        res = evaluate_model_on_puzzles(model, args.puzzles)
        print(json.dumps(res, indent=2))
    elif args.cmd == "train-puzzles":
        res = train_supervised_puzzles(
            graph=args.graph,
            parent_checkpoint=args.parent,
            train_file=args.train,
            val_file=args.val,
            output_path=args.out,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            shuffled=args.shuffled,
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "train-puzzle-rl":
        res = train_puzzle_rl_episodes(
            graph=args.graph,
            parent_checkpoint=args.parent,
            train_file=args.train,
            output_path=args.out,
            episodes=args.episodes,
            lr=args.lr,
            seed=args.seed,
            shuffled=args.shuffled,
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "build-puzzles-10k":
        manifest = build_puzzle_10k_dataset(
            csv_zst_path=args.csv,
            november_path=args.november,
            october_val_path=args.october_val,
            output_dir=args.out_dir,
            target_train_per_bucket=args.train_per_bucket,
            target_val_per_bucket=args.val_per_bucket,
            target_test_per_bucket=args.test_per_bucket,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2))
    elif args.cmd == "train-puzzles-10k":
        res = train_supervised_puzzle_10k(
            graph=args.graph,
            parent_checkpoint=args.parent,
            train_puzzle_path=args.train_puzzles,
            val_puzzle_path=args.val_puzzles,
            oct_val_path=args.oct_val,
            train_chess_path=args.train_chess,
            output_path=args.out,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            shuffled=args.shuffled,
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
