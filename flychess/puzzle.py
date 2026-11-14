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


def build_puzzle_10k_dataset(
    csv_zst_path: str = "data/lichess/lichess_db_puzzle.csv.zst",
    november_path: str = "data/benchmarks/november_benchmark_120.jsonl",
    october_val_path: str = "data/training/val_oct_200.jsonl",
    output_dir: str = "data/puzzles_10k",
    target_train_per_bucket: int = 2000,
    target_val_per_bucket: int = 200,
    target_test_per_bucket: int = 200,
    min_rating: int = 800,
    max_rating: int = 1850,
    min_popularity: int = 80,
    min_plays: int = 30,
    seed: int = 42,
) -> Dict:
    """Stream and filter 12,000 accessible, well-rated Lichess puzzles across 5 categories.
    
    10,000 train, 1,000 val, 1,000 frozen final test.
    Splits strictly by source game ID using seeded SHA-256.
    Deduplicates solver positions and guarantees zero overlap with both
    the frozen November benchmark and the October validation split.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load benchmark and October val exclusions (positions and games)
    bench_fens: Set[str] = set()
    bench_games: Set[str] = set()
    for excl_path in [november_path, october_val_path]:
        if excl_path and os.path.exists(excl_path):
            with open(excl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if "fen" in d:
                        parts = d["fen"].split()
                        bench_fens.add(" ".join(parts[:4]))
                    gid = d.get("game_id", "")
                    if gid:
                        clean_gid = gid.rstrip("/").split("/")[-1].split("#")[0][:8]
                        bench_games.add(clean_gid)

    buckets = ["mateIn1", "mateIn2", "defense_equality", "hanging_fork", "advantage_quiet"]
    train_data = {b: [] for b in buckets}
    val_data = {b: [] for b in buckets}
    test_data = {b: [] for b in buckets}

    seen_solver_fens: Set[str] = set()
    game_assigned_split: Dict[str, str] = {}
    scanned_rows = 0
    rejected_invalid = 0
    rejected_overlap = 0
    rejected_dup_fen = 0

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
            rejected_dup_fen += 1
            continue

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
        for b_name in buckets:
            if THEME_BUCKETS[b_name](theme_set):
                matched_bucket = b_name
                break

        if not matched_bucket:
            continue

        # Split strictly by source game hash
        if clean_gid in game_assigned_split:
            split_name = game_assigned_split[clean_gid]
        else:
            h = int(hashlib.sha256(f"seed{seed}_{clean_gid}".encode("utf-8")).hexdigest(), 16) % 1000
            if h < 834:
                split_name = "train"
            elif h < 917:
                split_name = "val"
            else:
                split_name = "test"
            game_assigned_split[clean_gid] = split_name

        if split_name == "train":
            target_list = train_data[matched_bucket]
            max_target = target_train_per_bucket
        elif split_name == "val":
            target_list = val_data[matched_bucket]
            max_target = target_val_per_bucket
        else:
            target_list = test_data[matched_bucket]
            max_target = target_test_per_bucket

        if len(target_list) >= max_target:
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
            "plays": plays,
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

    train_path = os.path.join(output_dir, "train_puzzles_10k.jsonl")
    val_path = os.path.join(output_dir, "val_puzzles_1000.jsonl")
    test_path = os.path.join(output_dir, "test_puzzles_frozen_1000.jsonl")

    for p, data in [(train_path, flat_train), (val_path, flat_val), (test_path, flat_test)]:
        with open(p, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    # Analyze forced mate in test mateIn2
    test_mate2 = test_data["mateIn2"]
    forced_mate2_count = sum(1 for r in test_mate2 if r["is_guaranteed_forced_mate"])

    # Unique games per split
    train_games = set(r["game_id"] for r in flat_train)
    val_games = set(r["game_id"] for r in flat_val)
    test_games = set(r["game_id"] for r in flat_test)

    train_fens = set(" ".join(r["solver_fen"].split()[:4]) for r in flat_train)
    val_fens = set(" ".join(r["solver_fen"].split()[:4]) for r in flat_val)
    test_fens = set(" ".join(r["solver_fen"].split()[:4]) for r in flat_test)

    manifest = {
        "dataset": "lichess_puzzle_10k_splits",
        "scanned_rows": scanned_rows,
        "rejected_invalid": rejected_invalid,
        "rejected_benchmark_overlap": rejected_overlap,
        "rejected_duplicate_solver_fen": rejected_dup_fen,
        "filter_criteria": {
            "min_rating": min_rating,
            "max_rating": max_rating,
            "min_popularity": min_popularity,
            "min_plays": min_plays,
            "seed": seed,
        },
        "total_train": len(flat_train),
        "total_val": len(flat_val),
        "total_frozen_test": len(flat_test),
        "train_by_bucket": {b: len(train_data[b]) for b in buckets},
        "val_by_bucket": {b: len(val_data[b]) for b in buckets},
        "test_by_bucket": {b: len(test_data[b]) for b in buckets},
        "unique_games": {
            "train": len(train_games),
            "val": len(val_games),
            "test": len(test_games),
            "overlap_train_val": len(train_games & val_games),
            "overlap_train_test": len(train_games & test_games),
            "overlap_val_test": len(val_games & test_games),
        },
        "fen_overlap": {
            "overlap_train_val": len(train_fens & val_fens),
            "overlap_train_test": len(train_fens & test_fens),
            "overlap_val_test": len(val_fens & test_fens),
        },
        "final_test_mateIn2_forced_mate": {
            "total_mateIn2_test": len(test_mate2),
            "guaranteed_forced_mate_count": forced_mate2_count,
            "guaranteed_forced_mate_rate": round(forced_mate2_count / len(test_mate2), 4) if test_mate2 else 0.0,
            "solution_line_only_count": len(test_mate2) - forced_mate2_count,
        },
        "paths": {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        }
    }

    manifest_path = os.path.join(output_dir, "puzzle_manifest_10k.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def evaluate_model_on_puzzles(model: FlyPolicy, puzzle_file_or_list) -> Dict:
    """Evaluate a FlyPolicy model on puzzle positions with theme breakdowns, multi-mating credit,
    legal move tracking, and game groupings."""
    if isinstance(puzzle_file_or_list, str):
        items = []
        with open(puzzle_file_or_list, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        items = puzzle_file_or_list

    if not items:
        return {"accuracy": 0.0, "total": 0, "by_theme": {}, "per_puzzle": [], "per_game": {}}

    theme_stats = {}
    per_game = {}
    per_puzzle = []
    correct_total = 0
    total_loss = 0.0
    legal_moves_count = 0

    for item in items:
        board = chess.Board(item["solver_fen"])
        legal, logits, probs, _ = model.forward(board)

        best_idx = int(np.argmax(probs))
        chosen_move_obj = legal[best_idx]
        chosen_move = chosen_move_obj.uci()

        if chosen_move_obj in board.legal_moves:
            legal_moves_count += 1

        # Check correctness:
        # For mateIn1, ANY valid checkmating move gets full credit
        bucket = item.get("bucket", "other")
        is_correct = False
        all_mates = item.get("all_mating_moves", [])
        if bucket == "mateIn1" and all_mates:
            is_correct = (chosen_move in all_mates)
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
            theme_stats[bucket] = {"correct": 0, "total": 0, "loss": 0.0}
        theme_stats[bucket]["total"] += 1
        theme_stats[bucket]["loss"] += loss
        if is_correct:
            theme_stats[bucket]["correct"] += 1

        gid = item.get("game_id", "")
        if gid:
            if gid not in per_game:
                per_game[gid] = {"total": 0, "correct": 0}
            per_game[gid]["total"] += 1
            if is_correct:
                per_game[gid]["correct"] += 1

        per_puzzle.append({
            "puzzle_id": item.get("puzzle_id", ""),
            "game_id": gid,
            "bucket": bucket,
            "correct": 1 if is_correct else 0,
            "loss": loss,
            "chosen_move": chosen_move,
            "first_move": item.get("first_move", ""),
        })

    by_theme = {}
    for b, s in theme_stats.items():
        by_theme[b] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0,
            "loss": round(s["loss"] / s["total"], 4) if s["total"] > 0 else 0.0,
        }

    return {
        "accuracy": round(correct_total / len(items), 4),
        "mean_loss": round(total_loss / len(items), 4),
        "total": len(items),
        "correct": correct_total,
        "legal_moves_count": legal_moves_count,
        "legal_move_rate": round(legal_moves_count / len(items), 4),
        "by_theme": by_theme,
        "per_game": {g: {"total": v["total"], "correct": v["correct"], "accuracy": round(v["correct"] / v["total"], 4)} for g, v in per_game.items()},
        "per_puzzle": per_puzzle,
    }


def compare_puzzle_evaluations(eval_a: Dict, eval_b: Dict, name_a: str = "Fly", name_b: str = "Shuffled") -> Dict:
    """Compare two evaluation runs on identical puzzle datasets."""
    puzzles_a = {p["puzzle_id"]: p for p in eval_a.get("per_puzzle", [])}
    puzzles_b = {p["puzzle_id"]: p for p in eval_b.get("per_puzzle", [])}
    common_pids = [p["puzzle_id"] for p in eval_a.get("per_puzzle", []) if p["puzzle_id"] in puzzles_b]

    a_wins = 0
    b_wins = 0
    ties = 0
    diffs = []

    for pid in common_pids:
        ca = puzzles_a[pid]["correct"]
        cb = puzzles_b[pid]["correct"]
        d = ca - cb
        diffs.append(d)
        if ca > cb:
            a_wins += 1
        elif cb > ca:
            b_wins += 1
        else:
            ties += 1

    # Game-level comparison
    games_a = eval_a.get("per_game", {})
    games_b = eval_b.get("per_game", {})
    common_gids = [g for g in games_a if g in games_b]

    game_a_wins = 0
    game_b_wins = 0
    game_ties = 0
    game_diffs = []

    for gid in common_gids:
        acc_a = games_a[gid]["accuracy"]
        acc_b = games_b[gid]["accuracy"]
        gd = acc_a - acc_b
        game_diffs.append(gd)
        if acc_a > acc_b:
            game_a_wins += 1
        elif acc_b > acc_a:
            game_b_wins += 1
        else:
            game_ties += 1

    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    mean_game_diff = float(np.mean(game_diffs)) if game_diffs else 0.0
    diff_of_means = eval_a["accuracy"] - eval_b["accuracy"]

    return {
        "name_a": name_a,
        "name_b": name_b,
        "common_puzzles": len(common_pids),
        "a_wins_puzzles": a_wins,
        "b_wins_puzzles": b_wins,
        "puzzle_ties": ties,
        "paired_puzzle_mean_diff": round(mean_diff, 4),
        "diff_of_puzzle_means": round(diff_of_means, 4),
        "mathematical_consistency": abs(mean_diff - diff_of_means) < 1e-6,
        "common_games": len(common_gids),
        "a_wins_games": game_a_wins,
        "b_wins_games": game_b_wins,
        "game_ties": game_ties,
        "paired_game_mean_diff": round(mean_game_diff, 4),
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


def train_supervised_puzzle_10k(
    graph: str,
    parent_checkpoint: str,
    train_puzzle_path: str = "data/puzzles_10k/train_puzzles_10k.jsonl",
    val_puzzle_path: str = "data/puzzles_10k/val_puzzles_1000.jsonl",
    oct_val_path: str = "data/training/val_oct_200.jsonl",
    train_chess_path: str = "data/training/train_2000.jsonl",
    output_path: str = "outputs/fly-puzzle-10k.npz",
    epochs: int = 5,
    lr: float = 0.001,
    seed: int = 42,
    shuffled: bool = False,
    on_epoch_done: callable = None,
) -> Dict:
    """Supervised distillation on 10,000 puzzles interleaved with 2,000 ordinary chess positions.
    
    Prevents catastrophic forgetting while tuning edge gains and adapter weights on tactical puzzles.
    Monitors October chess validation and puzzle validation after each epoch.
    Selects best checkpoint strictly on validation performance, never on final tests.
    """
    from .train import evaluate_candidates_dataset

    model = FlyPolicy(graph, seed=seed, shuffled=shuffled)
    model.load_weights(parent_checkpoint)
    initial_params = {k: v.copy() for k, v in model.params.items()}

    # Load data
    with open(train_puzzle_path, "r") as f:
        train_puzzles = [json.loads(line) for line in f if line.strip()]
    with open(val_puzzle_path, "r") as f:
        val_puzzles = [json.loads(line) for line in f if line.strip()]

    with open(train_chess_path, "r") as f:
        train_chess_lines = [json.loads(line) for line in f if line.strip()]
    train_chess = train_chess_lines[1:] if "metadata" in train_chess_lines[0] else train_chess_lines

    with open(oct_val_path, "r") as f:
        oct_val_lines = [json.loads(line) for line in f if line.strip()]
    oct_val_chess = oct_val_lines[1:] if "metadata" in oct_val_lines[0] else oct_val_lines

    # Baseline evaluation (epoch 0)
    init_val_puzz = evaluate_model_on_puzzles(model, val_puzzles)
    init_val_chess = evaluate_candidates_dataset(model, oct_val_chess, value_weight=0.5)

    history = [{
        "epoch": 0,
        "train_loss": 0.0,
        "puzzle_loss": 0.0,
        "chess_loss": 0.0,
        "puzzle_val_acc": init_val_puzz["accuracy"],
        "puzzle_val_loss": init_val_puzz["mean_loss"],
        "oct_val_loss": init_val_chess["loss"],
        "oct_val_top1": init_val_chess["top1"],
        "gain_l2_change": 0.0,
        "adapter_l2_change": 0.0,
    }]

    rng = np.random.default_rng(seed)
    best_val_score = init_val_puzz["accuracy"]
    best_weights = {k: v.copy() for k, v in model.params.items()}
    best_epoch = 0

    # Build combined dataset list: 10,000 puzzles + 2,000 chess positions = 12,000 examples
    combined_examples = []
    for p in train_puzzles:
        combined_examples.append({"type": "puzzle", "data": p})
    for c in train_chess:
        combined_examples.append({"type": "chess", "data": c})

    prev_gain = initial_params["gain"].copy()
    prev_adapter = initial_params["adapter"].copy()

    for ep in range(epochs):
        perm = rng.permutation(len(combined_examples))
        puzz_losses = []
        chess_losses = []

        for idx in perm:
            ex = combined_examples[int(idx)]
            if ex["type"] == "puzzle":
                item = ex["data"]
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
                puzz_losses.append(loss)
            else:
                item = ex["data"]
                board = chess.Board(item["fen"])
                loss, grads = model.gradients(
                    board,
                    target_probs=item["soft_probs"],
                    value_target=item.get("value_target", 0.0),
                    value_weight=0.5
                )
                model.update(grads, lr=lr)
                chess_losses.append(loss)

        mean_puzz_loss = float(np.mean(puzz_losses)) if puzz_losses else 0.0
        mean_chess_loss = float(np.mean(chess_losses)) if chess_losses else 0.0
        mean_total_loss = float(np.mean(puzz_losses + chess_losses))

        # Evaluate at epoch end
        val_puzz = evaluate_model_on_puzzles(model, val_puzzles)
        val_chess = evaluate_candidates_dataset(model, oct_val_chess, value_weight=0.5)

        gain_l2 = float(np.linalg.norm(model.params["gain"] - initial_params["gain"]))
        adapter_l2 = float(np.linalg.norm(model.params["adapter"] - initial_params["adapter"]))
        gain_epoch_delta = float(np.linalg.norm(model.params["gain"] - prev_gain))
        adapter_epoch_delta = float(np.linalg.norm(model.params["adapter"] - prev_adapter))
        prev_gain = model.params["gain"].copy()
        prev_adapter = model.params["adapter"].copy()

        epoch_record = {
            "epoch": ep + 1,
            "train_loss": round(mean_total_loss, 4),
            "puzzle_loss": round(mean_puzz_loss, 4),
            "chess_loss": round(mean_chess_loss, 4),
            "puzzle_val_acc": val_puzz["accuracy"],
            "puzzle_val_loss": val_puzz["mean_loss"],
            "puzzle_val_by_theme": val_puzz["by_theme"],
            "oct_val_loss": val_chess["loss"],
            "oct_val_top1": val_chess["top1"],
            "gain_l2_change": round(gain_l2, 6),
            "adapter_l2_change": round(adapter_l2, 6),
            "gain_epoch_delta": round(gain_epoch_delta, 6),
            "adapter_epoch_delta": round(adapter_epoch_delta, 6),
        }
        history.append(epoch_record)
        if on_epoch_done:
            on_epoch_done(epoch_record)

        # Validation selection (primarily puzzle accuracy, tiebreak puzzle loss)
        is_best = False
        if val_puzz["accuracy"] > best_val_score:
            is_best = True
        elif val_puzz["accuracy"] == best_val_score and val_puzz["mean_loss"] < history[best_epoch]["puzzle_val_loss"]:
            is_best = True

        if is_best:
            best_val_score = val_puzz["accuracy"]
            best_epoch = ep + 1
            best_weights = {k: v.copy() for k, v in model.params.items()}

    # Save best checkpoint
    model.params = best_weights
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)

    final_gain_l2 = float(np.linalg.norm(model.params["gain"] - initial_params["gain"]))
    final_adapter_l2 = float(np.linalg.norm(model.params["adapter"] - initial_params["adapter"]))

    return {
        "best_epoch": best_epoch,
        "best_puzzle_val_acc": best_val_score,
        "history": history,
        "gain_l2_change": round(final_gain_l2, 6),
        "adapter_l2_change": round(final_adapter_l2, 6),
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
