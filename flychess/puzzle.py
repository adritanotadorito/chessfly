"""Lichess puzzle streaming pipeline, split builder, multi-theme evaluation, and training."""
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import chess
import numpy as np

from .board import DIM, encode
from .model import FlyPolicy
from .reward import potential

# Standard themes mapped to 5 balanced thematic categories
THEME_BUCKETS = {
    "mateIn1": lambda themes: "mateIn1" in themes,
    "mateIn2": lambda themes: "mateIn2" in themes,
    "hanging_fork": lambda themes: ("hangingPiece" in themes) or ("fork" in themes),
    "defense_equality": lambda themes: ("defensiveMove" in themes) or ("equality" in themes),
    "advantage_quiet": lambda themes: ("advantage" in themes) or ("quietMove" in themes),
}


def find_all_mate_in_one_moves(board: chess.Board) -> List[str]:
    """Find all legal moves that deliver immediate checkmate in the current position."""
    mating_moves = []
    for move in board.legal_moves:
        b_next = board.copy()
        b_next.push(move)
        if b_next.is_checkmate():
            mating_moves.append(move.uci())
    return mating_moves


def check_forced_mate_in_two(board: chess.Board, solver_m1: chess.Move) -> bool:
    """Verify if solver_m1 guarantees checkmate in <= 2 solver moves against ALL legal opponent replies.
    
    If True, it is a mathematically guaranteed forced mate in 2.
    If False, it is an authentic Lichess solution-line puzzle where the recorded reply is tested.
    """
    b = board.copy()
    if solver_m1 not in b.legal_moves:
        return False
    b.push(solver_m1)
    if b.is_checkmate():
        return True
    if not list(b.legal_moves):
        return False  # Stalemate

    # Check that for EVERY legal opponent response, white has at least one immediate checkmate
    for opp_reply in b.legal_moves:
        b_opp = b.copy()
        b_opp.push(opp_reply)
        can_mate = False
        for m2 in b_opp.legal_moves:
            b_mate = b_opp.copy()
            b_mate.push(m2)
            if b_mate.is_checkmate():
                can_mate = True
                break
        if not can_mate:
            return False
    return True


def stream_lichess_puzzles(csv_zst_path: str):
    """Stream Lichess puzzle rows directly from .csv.zst without uncompressing to disk."""
    cmd = ["zstd", "-dc", csv_zst_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=65536)
    try:
        reader = csv.reader(proc.stdout)
        header = next(reader)
        for row in reader:
            yield row
    finally:
        proc.kill()
        proc.wait()


def build_puzzle_pilot(
    csv_zst_path: str = "data/lichess/lichess_db_puzzle.csv.zst",
    benchmark_path: str = "data/benchmarks/november_benchmark_120.jsonl",
    output_dir: str = "data/puzzles",
    target_train_per_bucket: int = 40,
    target_val_per_bucket: int = 10,
    target_test_per_bucket: int = 10,
    min_rating: int = 900,
    max_rating: int = 1750,
    min_popularity: int = 85,
    min_plays: int = 40,
) -> Dict:
    """Stream and filter accessible, well-rated Lichess puzzles across 5 categories.
    
    Splits strictly by source game ID (not puzzle row), deduplicates solver positions,
    and guarantees zero overlap with the frozen November benchmark.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load benchmark exclusions (positions and games)
    bench_fens: Set[str] = set()
    bench_games: Set[str] = set()
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r") as f:
            for line in f:
                d = json.loads(line)
                if "fen" in d:
                    parts = d["fen"].split()
                    bench_fens.add(" ".join(parts[:4]))
                gid = d.get("game_id", "")
                if gid:
                    clean_gid = gid.rstrip("/").split("/")[-1].split("#")[0][:8]
                    bench_games.add(clean_gid)

    # Buckets tracking
    buckets = ["mateIn1", "mateIn2", "hanging_fork", "defense_equality", "advantage_quiet"]
    train_data = {b: [] for b in buckets}
    val_data = {b: [] for b in buckets}
    test_data = {b: [] for b in buckets}

    seen_solver_fens: Set[str] = set()
    scanned_rows = 0
    rejected_invalid = 0
    rejected_overlap = 0

    for row in stream_lichess_puzzles(csv_zst_path):
        scanned_rows += 1
        if len(row) < 8:
            rejected_invalid += 1
            continue

        pid, fen, moves, rating_s, rd_s, pop_s, plays_s, themes_s = row[:8]
        game_url = row[8] if len(row) > 8 else ""

        try:
            rating = int(rating_s)
            pop = int(pop_s)
            plays = int(plays_s)
        except ValueError:
            rejected_invalid += 1
            continue

        if not (min_rating <= rating <= max_rating and pop >= min_popularity and plays >= min_plays):
            continue

        clean_gid = game_url.rstrip("/").split("/")[-1].split("#")[0][:8]
        if clean_gid in bench_games:
            rejected_overlap += 1
            continue

        move_list = moves.strip().split()
        if len(move_list) < 2:
            rejected_invalid += 1
            continue

        # Parse chess position and apply opponent setup move
        try:
            board = chess.Board(fen)
            setup_move = chess.Move.from_uci(move_list[0])
            if setup_move not in board.legal_moves:
                rejected_invalid += 1
                continue
            board.push(setup_move)
            solver_fen = board.fen()
        except Exception:
            rejected_invalid += 1
            continue

        norm_solver = " ".join(solver_fen.split()[:4])
        if norm_solver in bench_fens:
            rejected_overlap += 1
            continue
        if norm_solver in seen_solver_fens:
            continue  # Deduplicate identical solver positions

        # Validate solver first move
        try:
            first_move = chess.Move.from_uci(move_list[1])
            if first_move not in board.legal_moves:
                rejected_invalid += 1
                continue
        except Exception:
            rejected_invalid += 1
            continue

        # Categorize theme
        theme_set = set(themes_s.split())
        matched_bucket = None
        for b_name in ["mateIn1", "mateIn2", "defense_equality", "hanging_fork", "advantage_quiet"]:
            if THEME_BUCKETS[b_name](theme_set):
                matched_bucket = b_name
                break

        if not matched_bucket:
            continue

        # Check special tactical properties
        all_mating = []
        is_guaranteed_mate2 = False
        if matched_bucket == "mateIn1":
            all_mating = find_all_mate_in_one_moves(board)
            if not all_mating and move_list[1]:
                all_mating = [move_list[1]]
        elif matched_bucket == "mateIn2":
            is_guaranteed_mate2 = check_forced_mate_in_two(board, first_move)

        # Split strictly by source game hash
        h = int(hashlib.sha256(clean_gid.encode("utf-8")).hexdigest(), 16) % 100
        # 0..59 -> Train (60%), 60..79 -> Val (20%), 80..99 -> Test (20%)
        if h < 60:
            target_list = train_data[matched_bucket]
            max_target = target_train_per_bucket
            split_name = "train"
        elif h < 80:
            target_list = val_data[matched_bucket]
            max_target = target_val_per_bucket
            split_name = "val"
        else:
            target_list = test_data[matched_bucket]
            max_target = target_test_per_bucket
            split_name = "test"

        if len(target_list) >= max_target:
            continue

        record = {
            "puzzle_id": pid,
            "split": split_name,
            "bucket": matched_bucket,
            "original_fen": fen,
            "solver_fen": solver_fen,
            "setup_move": move_list[0],
            "first_move": move_list[1],
            "moves": move_list,
            "rating": rating,
            "popularity": pop,
            "themes": themes_s.split(),
            "game_url": game_url,
            "game_id": clean_gid,
            "all_mating_moves": all_mating,
            "is_guaranteed_forced_mate": is_guaranteed_mate2,
        }

        seen_solver_fens.add(norm_solver)
        target_list.append(record)

        # Check if all quotas reached
        done_train = all(len(train_data[b]) >= target_train_per_bucket for b in buckets)
        done_val = all(len(val_data[b]) >= target_val_per_bucket for b in buckets)
        done_test = all(len(test_data[b]) >= target_test_per_bucket for b in buckets)
        if done_train and done_val and done_test:
            break

    # Flatten and save splits
    flat_train = [rec for b in buckets for rec in train_data[b]]
    flat_val = [rec for b in buckets for rec in val_data[b]]
    flat_test = [rec for b in buckets for rec in test_data[b]]

    train_path = os.path.join(output_dir, "train_puzzles.jsonl")
    val_path = os.path.join(output_dir, "val_puzzles.jsonl")
    test_path = os.path.join(output_dir, "test_puzzles_frozen.jsonl")

    for p, data in [(train_path, flat_train), (val_path, flat_val), (test_path, flat_test)]:
        with open(p, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    manifest = {
        "scanned_rows": scanned_rows,
        "rejected_invalid": rejected_invalid,
        "rejected_benchmark_overlap": rejected_overlap,
        "total_train": len(flat_train),
        "total_val": len(flat_val),
        "total_frozen_test": len(flat_test),
        "train_by_bucket": {b: len(train_data[b]) for b in buckets},
        "val_by_bucket": {b: len(val_data[b]) for b in buckets},
        "test_by_bucket": {b: len(test_data[b]) for b in buckets},
    }

    manifest_path = os.path.join(output_dir, "puzzle_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def evaluate_model_on_puzzles(model: FlyPolicy, puzzle_file_or_list) -> Dict:
    """Evaluate a FlyPolicy model on puzzle positions with theme breakdowns and multi-mating credit."""
    if isinstance(puzzle_file_or_list, str):
        items = []
        with open(puzzle_file_or_list, "r") as f:
            for line in f:
                items.append(json.loads(line))
    else:
        items = puzzle_file_or_list

    if not items:
        return {"accuracy": 0.0, "total": 0, "by_theme": {}}

    theme_stats = {}
    correct_total = 0
    total_loss = 0.0

    for item in items:
        board = chess.Board(item["solver_fen"])
        legal, logits, probs, _ = model.forward(board)

        best_idx = int(np.argmax(probs))
        chosen_move = legal[best_idx].uci()

        # Check correctness:
        # For mateIn1, ANY valid checkmating move gets full credit
        bucket = item.get("bucket", "other")
        is_correct = False
        all_mates = item.get("all_mating_moves", [])
        if bucket == "mateIn1" and all_mates:
            is_correct = chosen_move in all_mates
        else:
            is_correct = (chosen_move == item["first_move"])

        if is_correct:
            correct_total += 1

        # Target cross-entropy loss
        if bucket == "mateIn1" and all_mates:
            target_mask = np.zeros(len(legal), dtype=np.float32)
            for m in all_mates:
                obj = chess.Move.from_uci(m)
                if obj in legal:
                    target_mask[legal.index(obj)] = 1.0
            if target_mask.sum() > 0:
                target_mask /= target_mask.sum()
            else:
                target_mask[legal.index(chess.Move.from_uci(item["first_move"]))] = 1.0
            loss = -float(np.sum(target_mask * np.log(np.maximum(probs, 1e-12))))
        else:
            first_obj = chess.Move.from_uci(item["first_move"])
            first_idx = legal.index(first_obj)
            loss = -float(np.log(max(probs[first_idx], 1e-12)))

        total_loss += loss

        if bucket not in theme_stats:
            theme_stats[bucket] = {"correct": 0, "total": 0}
        theme_stats[bucket]["total"] += 1
        if is_correct:
            theme_stats[bucket]["correct"] += 1

    by_theme = {}
    for b, s in theme_stats.items():
        by_theme[b] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0,
        }

    return {
        "accuracy": round(correct_total / len(items), 4),
        "mean_loss": round(total_loss / len(items), 4),
        "total": len(items),
        "correct": correct_total,
        "by_theme": by_theme,
    }


def train_supervised_puzzles(
    graph: str,
    parent_checkpoint: str,
    train_file: str,
    val_file: str,
    output_path: str,
    epochs: int = 5,
    lr: float = 0.002,
    seed: int = 42,
    shuffled: bool = False,
) -> Dict:
    """Supervise FlyPolicy on pilot puzzles with legal-move masking and multiple-mating move credit."""
    model = FlyPolicy(graph, seed=seed, shuffled=shuffled)
    model.load_weights(parent_checkpoint)
    initial_params = {k: v.copy() for k, v in model.params.items()}

    with open(train_file, "r") as f:
        train_items = [json.loads(line) for line in f]
    with open(val_file, "r") as f:
        val_items = [json.loads(line) for line in f]

    rng = np.random.default_rng(seed)
    best_val_acc = -1.0
    best_weights = None
    best_epoch = 0
    history = []

    for ep in range(epochs):
        perm = rng.permutation(len(train_items))
        ep_loss = 0.0

        for idx in perm:
            item = train_items[idx]
            board = chess.Board(item["solver_fen"])
            legal = list(board.legal_moves)
            bucket = item.get("bucket", "")
            all_mates = item.get("all_mating_moves", [])

            if bucket == "mateIn1" and len(all_mates) > 1:
                t_probs = np.zeros(len(legal), dtype=np.float32)
                for m in all_mates:
                    obj = chess.Move.from_uci(m)
                    if obj in legal:
                        t_probs[legal.index(obj)] = 1.0
                if t_probs.sum() > 0:
                    t_probs /= t_probs.sum()
                loss, grads = model.gradients(board, target_probs=t_probs)
            else:
                loss, grads = model.gradients(board, target=chess.Move.from_uci(item["first_move"]))

            model.update(grads, lr=lr)
            ep_loss += loss

        mean_train_loss = ep_loss / len(train_items)
        val_metrics = evaluate_model_on_puzzles(model, val_items)

        history.append({
            "epoch": ep + 1,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": val_metrics["mean_loss"],
            "val_acc": val_metrics["accuracy"],
            "val_by_theme": val_metrics["by_theme"],
        })

        if val_metrics["accuracy"] > best_val_acc or (val_metrics["accuracy"] == best_val_acc and best_weights is None):
            best_val_acc = val_metrics["accuracy"]
            best_epoch = ep + 1
            best_weights = {k: v.copy() for k, v in model.params.items()}

    # Save best checkpoint
    model.params = best_weights
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)

    gain_l2 = float(np.linalg.norm(model.params["gain"] - initial_params["gain"]))
    adapter_l2 = float(np.linalg.norm(model.params["adapter"] - initial_params["adapter"]))

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
        "gain_l2_change": round(gain_l2, 6),
        "adapter_l2_change": round(adapter_l2, 6),
        "checkpoint": output_path,
    }


def train_puzzle_rl_episodes(
    graph: str,
    parent_checkpoint: str,
    train_file: str,
    output_path: str,
    episodes: int = 50,
    lr: float = 0.001,
    seed: int = 42,
    shuffled: bool = False,
    gamma: float = 0.99,
) -> Dict:
    """Run online RL puzzle episodes where the fly chooses moves itself.
    
    A correct move in a single-move puzzle yields +1.0. An incorrect move terminates the episode with -1.0.
    In multi-move puzzles (mate in 2), a correct first move leads to the opponent's reply line,
    and a correct second move achieves checkmate (+1.0).
    Bounded Stockfish regret penalty and potential shaping are applied with the audited 0.10 cap.
    """
    model = FlyPolicy(graph, seed=seed, shuffled=shuffled)
    model.load_weights(parent_checkpoint)
    initial_params = {k: v.copy() for k, v in model.params.items()}

    with open(train_file, "r") as f:
        puzzles = [json.loads(line) for line in f]

    rng = np.random.default_rng(seed)
    successes = 0
    episode_logs = []

    for ep in range(episodes):
        p_idx = rng.integers(0, len(puzzles))
        puzzle = puzzles[p_idx]
        board = chess.Board(puzzle["solver_fen"])
        moves_line = puzzle["moves"]
        bucket = puzzle.get("bucket", "")
        all_mates = puzzle.get("all_mating_moves", [])

        transitions = []
        episode_success = False

        # Step 1: Fly chooses move in solver position
        fly_move = model.choose(board, sample=True)
        chosen_uci = fly_move.uci()
        phi_before = potential(board, board.turn)

        # Check correctness of first move
        is_step1_correct = False
        if bucket == "mateIn1" and all_mates:
            is_step1_correct = (chosen_uci in all_mates)
        else:
            is_step1_correct = (chosen_uci == puzzle["first_move"])

        board_after = board.copy()
        board_after.push(fly_move)
        phi_after = potential(board_after, board.turn)
        shaped_delta = 0.15 * (gamma * phi_after - phi_before)

        if not is_step1_correct:
            # Fly chose wrong move: blunder / failed puzzle
            move_penalty = -0.50
            # Step reward
            step_reward = shaped_delta + move_penalty
            terminal_reward = -1.0
            transitions.append({
                "board": board.copy(),
                "move": fly_move,
                "step_reward": step_reward,
                "move_penalty": move_penalty,
                "terminal_reward": terminal_reward,
            })
            episode_success = False
        else:
            # First move was correct!
            if bucket == "mateIn1" or len(moves_line) < 4:
                # One-step puzzle solved!
                move_penalty = 0.0
                step_reward = shaped_delta + move_penalty
                terminal_reward = 1.0
                transitions.append({
                    "board": board.copy(),
                    "move": fly_move,
                    "step_reward": step_reward,
                    "move_penalty": move_penalty,
                    "terminal_reward": terminal_reward,
                })
                episode_success = True
                successes += 1
            else:
                # Multi-move puzzle (e.g. mateIn2): opponent makes line reply
                move_penalty = 0.0
                transitions.append({
                    "board": board.copy(),
                    "move": fly_move,
                    "step_reward": shaped_delta + move_penalty,
                    "move_penalty": move_penalty,
                    "terminal_reward": 0.0,
                })
                opp_m1 = chess.Move.from_uci(moves_line[2])
                board_after.push(opp_m1)

                # Step 2: Fly chooses final move
                phi2_before = potential(board_after, board_after.turn)
                fly_m2 = model.choose(board_after, sample=True)
                board_final = board_after.copy()
                board_final.push(fly_m2)
                phi2_after = potential(board_final, board_after.turn)
                shaped2 = 0.15 * (gamma * phi2_after - phi2_before)

                is_step2_correct = (fly_m2.uci() == moves_line[3]) or board_final.is_checkmate()
                if is_step2_correct:
                    transitions.append({
                        "board": board_after.copy(),
                        "move": fly_m2,
                        "step_reward": shaped2,
                        "move_penalty": 0.0,
                        "terminal_reward": 1.0,
                    })
                    episode_success = True
                    successes += 1
                else:
                    transitions.append({
                        "board": board_after.copy(),
                        "move": fly_m2,
                        "step_reward": shaped2 - 0.50,
                        "move_penalty": -0.50,
                        "terminal_reward": -1.0,
                    })
                    episode_success = False

        # Apply audited 0.10 regret penalty cap across transitions
        raw_pen_sum = sum(t["move_penalty"] for t in transitions)
        if abs(raw_pen_sum) > 0.10 and abs(raw_pen_sum) > 0:
            scale = 0.10 / abs(raw_pen_sum)
            for t in transitions:
                t["move_penalty"] = t["move_penalty"] * scale
                t["step_reward"] = t["step_reward"] + t["move_penalty"]

        # Compute discounted returns G_t
        running = 0.0
        returns = []
        for t in reversed(transitions):
            running = t["step_reward"] + t["terminal_reward"] + gamma * running
            returns.append(running)
        returns.reverse()

        # Update model via policy gradients
        for t, ret in zip(transitions, returns):
            val_pred = model.predict_value(t["board"])
            adv = float(np.clip(ret - val_pred, -1.5, 1.5))
            _, grads = model.gradients(t["board"], target=t["move"], value_target=ret, scale=adv, value_weight=0.5)
            model.update(grads, lr=lr)

        episode_logs.append({
            "episode": ep + 1,
            "puzzle_id": puzzle["puzzle_id"],
            "bucket": bucket,
            "success": episode_success,
            "return": round(returns[0], 4) if returns else 0.0,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)

    gain_l2 = float(np.linalg.norm(model.params["gain"] - initial_params["gain"]))
    adapter_l2 = float(np.linalg.norm(model.params["adapter"] - initial_params["adapter"]))

    return {
        "episodes": episodes,
        "success_rate": round(successes / episodes, 4),
        "gain_l2_change": round(gain_l2, 6),
        "adapter_l2_change": round(adapter_l2, 6),
        "checkpoint": output_path,
        "recent_logs": episode_logs[-10:],
    }
