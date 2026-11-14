"""Local ChessFly dashboard. Comprehensive GUI for Bullet matches, RL training, and SFT."""
import json
import shutil
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.svg
import numpy as np
import streamlit as st

from flychess.graph import build_graph, load_graph
from flychess.match import BulletMatch, format_clock
from flychess.model import FlyPolicy, summarize
from flychess.reward import compute_game_rewards, features, potential
from flychess.train import (accuracy, evaluate_bullet_matches, evaluate_heldout,
                            generate, read_examples, train_rl_bullet,
                            train_supervised)

st.set_page_config(page_title="ChessFly Lab", page_icon="🪰", layout="wide")
st.title("🪰 ChessFly Lab: Fly-Wired Chess Policy")
st.caption("A connectome-constrained chess agent whose recurrent connections follow real FlyWire Drosophila synapses.")

ROOT = Path(__file__).resolve().parent


def resolved(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def status(label, ok, detail=""):
    st.markdown(f"{'✅' if ok else '○'} **{label}**  {detail}")


with st.sidebar:
    st.header("Project Configuration")
    graph_str = st.text_input("FlyWire graph", "data/flywire-subgraph.npz")
    dataset_str = st.text_input("Teacher positions (SFT)", "data/teacher-200.jsonl")
    fly_str = st.text_input("Fly checkpoint (SFT)", "outputs/fly-200.npz")
    shuffle_str = st.text_input("Shuffled checkpoint (SFT)", "outputs/shuffled-200.npz")
    rl_fly_str = st.text_input("Fly RL checkpoint", "outputs/fly-rl-bullet.npz")
    stockfish_str = st.text_input("Stockfish executable", shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    st.caption("Paths are relative to the chessfly folder.")

graph_path = resolved(graph_str)
dataset_path = resolved(dataset_str)
fly_path = resolved(fly_str)
shuffle_path = resolved(shuffle_str)
rl_fly_path = resolved(rl_fly_str)
stockfish_path = resolved(stockfish_str)

meta = None
nodes = 0
if graph_path.is_file():
    try:
        _, _, _, meta, nodes = load_graph(str(graph_path))
    except Exception as exc:
        st.error(f"Cannot read graph: {exc}")

overview, board_tab, bullet_tab, rl_tab, sft_tab, results_tab = st.tabs([
    "Overview & Science",
    "Interactive Board",
    "⚡ Bullet Match (1+0)",
    "🧠 Reward-Based RL",
    "SFT / Move Imitation",
    "Compare Results",
])

with overview:
    st.subheader("Experimental Framework")
    st.markdown("""
    **1 · Authentic Connectome Topology.** We extract a 256-neuron, 396-synapse connected subgraph
    from the real FlyWire (*Drosophila melanogaster*) whole-brain connectome (FAFB v783).
    The dense board adapter (785 features) and move readouts are engineered; the internal recurrent
    topology strictly follows the biological wiring diagram.

    **2 · Supervised Warm Start (SFT).** The model initializes by learning teacher moves from Stockfish.
    *This is supervised imitation, not a bullet match or reinforcement learning.*

    **3 · Real 1+0 Bullet Matches.** The model plays genuine 1-minute, zero-increment bullet games
    against a fixed Stockfish bot. Inference and engine think-times are billed strictly via `time.perf_counter()`.
    Flagging (loss on time) and insufficient material rules are strictly enforced.

    **4 · Transparent Chess Rewards.** Rather than opaque black-box signals, the agent receives
    an interpretable reward accounting for material, safe mobility, coordination, hanging pieces,
    passed pawns, king safety, and a bounded post-game Stockfish move regret penalty ($\le 0.10$).
    Match outcome (+1 win, 0 draw, -1 loss) remains dominant.

    **5 · Degree-Preserving Shuffled Control.** A matched network with identical in-degree and out-degree
    distributions and normalized synaptic scales serves as the empirical control.
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        status("Real FlyWire Graph", bool(meta and meta.get("authentic_flywire")),
               f"{nodes} neurons, {meta.get('edges', 0)} edges" if meta else "Not found")
    with c2:
        status("Stockfish Engine", Path(stockfish_str).is_file(),
               Path(stockfish_str).name if Path(stockfish_str).is_file() else "Not found")
    with c3:
        status("SFT Checkpoints", fly_path.is_file() and shuffle_path.is_file(),
               "200-game warm start" if fly_path.is_file() else "Pending")
    with c4:
        status("RL Bullet Checkpoint", rl_fly_path.is_file(),
               rl_fly_path.name if rl_fly_path.is_file() else "Pending")

    if meta:
        with st.expander("Connectome Provenance & Hash"):
            st.json(meta)

with board_tab:
    st.subheader("Explore Model Candidate Moves")
    if "board_fen" not in st.session_state:
        st.session_state.board_fen = chess.STARTING_FEN

    col_btn1, col_btn2 = st.columns([1, 5])
    if col_btn1.button("Reset to Start"):
        st.session_state.board_fen = chess.STARTING_FEN

    try:
        board = chess.Board(st.session_state.board_fen)
    except ValueError:
        board = chess.Board()
        st.session_state.board_fen = board.fen()

    left, right = st.columns([1, 1])
    with left:
        st.image(chess.svg.board(board=board, size=460), width=460)
        st.caption(f"{'White' if board.turn == chess.WHITE else 'Black'} to move · Move {board.fullmove_number}")

    with right:
        choices = {"Untrained": None}
        if fly_path.is_file():
            choices["FlyWire · SFT (fly-200.npz)"] = (fly_path, False)
        if rl_fly_path.is_file():
            choices["FlyWire · RL Bullet (fly-rl-bullet.npz)"] = (rl_fly_path, False)
        if shuffle_path.is_file():
            choices["Shuffled Control · SFT (shuffled-200.npz)"] = (shuffle_path, True)

        selected_model_name = st.selectbox("Select Policy Model", list(choices.keys()))
        selected_model_info = choices[selected_model_name]

        model = None
        if graph_path.is_file():
            try:
                shuf_flag = selected_model_info[1] if selected_model_info else False
                model = FlyPolicy(str(graph_path), shuffled=shuf_flag)
                if selected_model_info:
                    model.load_weights(str(selected_model_info[0]))

                if not board.is_game_over(claim_draw=True):
                    st.write("**Top Legal Policy Move Probabilities**")
                    top_moves = summarize(board, model, top=5)
                    for uci_move, prob in top_moves:
                        move_obj = chess.Move.from_uci(uci_move)
                        st.write(f"- `{board.san(move_obj)}` (`{uci_move}`): **{prob:.1%}**")
            except Exception as exc:
                st.error(f"Cannot load model: {exc}")

        if not board.is_game_over(claim_draw=True):
            legal = list(board.legal_moves)
            picked = st.selectbox("Play a move manually", legal,
                                  format_func=lambda m: f"{board.san(m)} ({m.uci()})")
            c_a, c_b = st.columns(2)
            if c_a.button("Play selected"):
                board.push(picked)
                st.session_state.board_fen = board.fen()
                st.rerun()
            if c_b.button("Let policy play move", disabled=model is None):
                board.push(model.choose(board))
                st.session_state.board_fen = board.fen()
                st.rerun()

with bullet_tab:
    st.subheader("⚡ Real 1+0 Bullet Match")
    st.markdown("Play **1 minute per side, zero increment** against Stockfish. Clocks are timed with `time.perf_counter()`; flagging and draw rules are enforced strictly.")

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        match_model_choice = st.selectbox("Policy Checkpoint", [
            ("outputs/fly-rl-bullet.npz", False) if rl_fly_path.is_file() else None,
            ("outputs/fly-200.npz", False) if fly_path.is_file() else None,
            ("outputs/shuffled-200.npz", True) if shuffle_path.is_file() else None,
        ], format_func=lambda x: x[0] if x else "None available")
    with b_col2:
        fly_color_choice = st.selectbox("Fly Model Color", ["White", "Black"])
    with b_col3:
        bot_diff_choice = st.selectbox("Stockfish Difficulty", ["weak", "medium", "strong"],
                                       format_func=lambda x: {
                                           "weak": "Weak (Skill 0, Elo 1350)",
                                           "medium": "Medium (Skill 5, Elo 1600)",
                                           "strong": "Strong (Skill 20, Full Strength)"
                                       }[x])

    start_match = st.button("▶ Start 1+0 Bullet Match", type="primary", disabled=match_model_choice is None or not Path(stockfish_str).is_file())

    if start_match:
        with st.spinner("Playing 1+0 bullet match against Stockfish…"):
            try:
                ckpt_path, is_shuf = match_model_choice
                m_policy = FlyPolicy(str(graph_path), shuffled=is_shuf)
                m_policy.load_weights(str(resolved(ckpt_path)))

                match = BulletMatch(
                    fly_model=m_policy,
                    stockfish_path=str(stockfish_path),
                    fly_color=chess.WHITE if fly_color_choice == "White" else chess.BLACK,
                    base_time=60.0,
                    increment=0.0,
                    bot_difficulty=bot_diff_choice,
                    sample_moves=False,
                )
                match_result = match.run()

                # Post-game transparent rewards
                engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
                try:
                    rewards = compute_game_rewards(match_result, engine=engine)
                    match_result["rewards"] = rewards
                finally:
                    engine.quit()

                st.session_state["last_bullet_match"] = match_result
                st.success(f"Match complete: {match_result['result']} ({match_result['termination_reason']})")
            except Exception as exc:
                st.error(f"Error during bullet match: {exc}")

    if "last_bullet_match" in st.session_state:
        m_data = st.session_state["last_bullet_match"]
        moves = m_data.get("moves", [])

        st.divider()
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        col_res1.metric("Result", m_data["result"])
        col_res2.metric("Winner", m_data["winner"].capitalize() if m_data["winner"] else "Draw")
        col_res3.metric("Termination", m_data["termination_reason"])
        col_res4.metric("Total Plies", m_data["ply_count"])

        # Clocks
        clock_c1, clock_c2 = st.columns(2)
        clock_c1.metric("⚪ White Clock", format_clock(m_data["white_clock"]),
                        help=f"Remaining: {m_data['white_clock']}s")
        clock_c2.metric("⚫ Black Clock", format_clock(m_data["black_clock"]),
                        help=f"Remaining: {m_data['black_clock']}s")

        # Replay Board
        replay_step = st.slider("Replay Move", 0, len(moves), len(moves))
        replay_board = chess.Board()
        for idx in range(replay_step):
            replay_board.push(chess.Move.from_uci(moves[idx]["uci"]))

        r_left, r_right = st.columns([1, 1])
        with r_left:
            st.image(chess.svg.board(board=replay_board, size=460), width=460)
            if replay_step > 0:
                cur_move = moves[replay_step - 1]
                st.caption(f"Ply {cur_move['ply']} · {cur_move['player']} played **{cur_move['san']}** ({cur_move['elapsed_ms']}ms)")

        with r_right:
            st.subheader("Transparent Reward Breakdown")
            if "rewards" in m_data:
                diag = m_data["rewards"]["diagnostics"]
                st.write(f"- **Final Match Outcome**: `{diag['terminal_reward']:+.1f}` ({diag['final_result']})")
                st.write(f"- **Total Training Reward**: `{diag['total_training_reward']:+.4f}`")
                st.write(f"- **Mean Centipawn Loss (ACPL)**: `{diag['mean_centipawn_loss']} cp`")
                st.write(f"- **Blunders (>300 cp)**: `{diag['blunder_count']}`")
                st.write(f"- **Acceptable Moves (≤50 cp)**: `{diag['acceptable_count']}`")
                st.write(f"- **Bounded Stockfish Regret Penalty**: `{diag['engine_penalty']:.4f}`")

                with st.expander("Detailed Move-by-Move Transition Breakdown"):
                    trans_rows = []
                    for t in m_data["rewards"]["transitions"]:
                        trans_rows.append({
                            "Ply": t["ply"],
                            "Move": t["san"],
                            "Φ Before": t["phi_before"],
                            "Φ After": t["phi_after"],
                            "ΔΦ Shaping": t["shaped_reward"],
                            "Engine Loss (cp)": t["loss_cp"],
                            "Net Step Reward": t["training_reward"],
                        })
                    st.dataframe(trans_rows, hide_index=True)

        with st.expander("Full Game PGN"):
            st.text_area("PGN", m_data["pgn"], height=200)

with rl_tab:
    st.subheader("🧠 Reward-Based Reinforcement Learning")
    st.markdown("""
    Online policy-gradient updates (**REINFORCE**) from complete 1+0 bullet games.
    The policy samples moves, plays against Stockfish, receives the transparent chess reward + terminal outcome,
    and updates **the 396 biological FlyWire edge gains** as well as the dense adapter and move readout.
    """)

    rl_c1, rl_c2, rl_c3 = st.columns(3)
    with rl_c1:
        rl_source = st.selectbox("Warm Start Policy", [
            str(fly_path),
            str(shuffle_path),
        ], format_func=lambda x: Path(x).name)
    with rl_c2:
        rl_matches = st.number_input("Matches to Play", min_value=2, max_value=20, value=6, step=2)
    with rl_c3:
        rl_out_str = st.text_input("New RL Checkpoint Path", "outputs/fly-rl-bullet.npz")

    start_rl = st.button("🚀 Start Reward-Based RL Training", type="primary")

    if start_rl:
        rl_out_path = resolved(rl_out_str)
        ensure_parent(rl_out_path)
        with st.spinner(f"Running RL training across {rl_matches} bullet games…"):
            try:
                rl_report = train_rl_bullet(
                    graph=str(graph_path),
                    checkpoint=str(resolved(rl_source)),
                    output=str(rl_out_path),
                    matches=int(rl_matches),
                    stockfish_path=str(stockfish_path),
                    bot_difficulty="weak",
                    lr=0.0005,
                )
                st.session_state["last_rl_report"] = rl_report
                st.success(f"RL training complete! Saved to {rl_out_path.name}")
            except Exception as exc:
                st.error(f"RL Training failed: {exc}")

    if "last_rl_report" in st.session_state:
        rep = st.session_state["last_rl_report"]
        st.subheader("Training Match Log")
        st.dataframe(rep["history"], hide_index=True)

        st.subheader("Verified Parameter Updates")
        st.caption("Demonstrating that weights—especially the 396 recurrent fly gains—were actually updated by chess rewards.")
        ch_rows = []
        for param_k, diffs in rep["parameter_changes"].items():
            ch_rows.append({
                "Parameter Group": param_k,
                "L2 Norm Change": diffs["l2_norm"],
                "Max Abs Change": diffs["max_abs"],
                "Mean Abs Change": diffs["mean_abs"],
            })
        st.dataframe(ch_rows, hide_index=True)

with sft_tab:
    st.subheader("SFT / Move Imitation (Stockfish Teacher Examples)")
    st.info("⚠️ This tab is for supervised imitation learning on Stockfish-generated dataset positions. This is preliminary move prediction, NOT bullet win rate.")

    sft_g1, sft_g2 = st.columns(2)
    with sft_g1:
        st.markdown("**1 · Generate Teacher Examples**")
        gen_games = st.number_input("Number of Games to Sample", min_value=10, max_value=1000, value=200, step=20)
        gen_out = st.text_input("Dataset output path", "data/teacher-200.jsonl")
        if st.button("Generate Teacher Moves"):
            try:
                with st.spinner("Generating positions…"):
                    rep = generate(str(resolved(gen_out)), str(stockfish_path), games=int(gen_games))
                st.success("Generated dataset!")
                st.json(rep)
            except Exception as exc:
                st.error(str(exc))

    with sft_g2:
        st.markdown("**2 · Supervised Training**")
        sft_dataset = st.text_input("Training Dataset", "data/teacher-200.jsonl")
        sft_epochs = st.number_input("SFT Epochs", min_value=1, max_value=20, value=8)
        sft_out_prefix = st.text_input("SFT Output Prefix", "outputs/sft-experiment")
        if st.button("Train SFT Fly + Shuffled"):
            try:
                ensure_parent(resolved(sft_out_prefix))
                fly_dest = f"{sft_out_prefix}-fly.npz"
                shuf_dest = f"{sft_out_prefix}-shuffled.npz"
                with st.spinner("Training supervised models with best-epoch checkpointing…"):
                    f_rep = train_supervised(str(graph_path), str(resolved(sft_dataset)), fly_dest, epochs=int(sft_epochs))
                    s_rep = train_supervised(str(graph_path), str(resolved(sft_dataset)), shuf_dest, epochs=int(sft_epochs), shuffled=True)
                st.success("SFT Complete!")
                st.json({
                    "fly_best_top1": f_rep["best_heldout"]["top1"],
                    "shuffled_best_top1": s_rep["best_heldout"]["top1"],
                })
            except Exception as exc:
                st.error(str(exc))

with results_tab:
    st.subheader("Rigorous Experimental Comparisons")

    st.markdown("#### A · Held-Out Move Imitation (SFT)")
    if dataset_path.is_file():
        try:
            _, examples = read_examples(str(dataset_path))
            held = [r for r in examples if r["split"] == "heldout"]
            st.write(f"Evaluating across **{len(held)} held-out positions**:")

            sft_rows = []
            for lbl, ckpt, shuf in [
                ("FlyWire · Untrained", None, False),
                ("FlyWire · SFT (fly-200.npz)", fly_path, False),
                ("Shuffled Control · Untrained", None, True),
                ("Shuffled Control · SFT (shuffled-200.npz)", shuffle_path, True),
            ]:
                if ckpt is not None and not ckpt.is_file():
                    continue
                pol = FlyPolicy(str(graph_path), shuffled=shuf)
                if ckpt:
                    pol.load_weights(str(ckpt))
                eval_metrics = evaluate_heldout(pol, held)
                sft_rows.append({
                    "Model": lbl,
                    "Top-1 Match": f"{eval_metrics['top1']:.1%}",
                    "Top-3 Match": f"{eval_metrics['top3']:.1%}",
                    "Top-5 Match": f"{eval_metrics['top5']:.1%}",
                    "Validation Loss": eval_metrics["loss"],
                })
            st.dataframe(sft_rows, hide_index=True)
        except Exception as exc:
            st.error(f"Error computing SFT comparisons: {exc}")

    st.markdown("#### B · Bullet Match Matchup Evaluation")
    st.caption("Fresh 1+0 bullet matches alternating colors against Stockfish.")

    run_bullet_eval = st.button("Run Matched Bullet Evaluation (4 Games Each)")
    if run_bullet_eval and Path(stockfish_str).is_file():
        with st.spinner("Running matched 4-game bullet evaluation for Pre-RL vs Post-RL vs Control…"):
            try:
                eval_models = [
                    ("Pre-RL (fly-200.npz)", str(fly_path), False),
                    ("Post-RL (fly-rl-bullet.npz)", str(rl_fly_path), False) if rl_fly_path.is_file() else None,
                    ("Shuffled Control (shuffled-200.npz)", str(shuffle_path), True),
                ]
                comp_rows = []
                for entry in eval_models:
                    if entry is None or not Path(entry[1]).is_file():
                        continue
                    name, ckpt_p, is_shuf = entry
                    p_model = FlyPolicy(str(graph_path), shuffled=is_shuf)
                    p_model.load_weights(ckpt_p)
                    eval_res = evaluate_bullet_matches(p_model, stockfish_path=str(stockfish_path), num_games=4, bot_difficulty="weak")
                    comp_rows.append({
                        "Model": name,
                        "Win / Draw / Loss": f"{eval_res['wins']} / {eval_res['draws']} / {eval_res['losses']}",
                        "Win Rate": f"{eval_res['win_rate']:.1%}",
                        "Time Losses (Flags)": eval_res["flags_lost"],
                        "Avg Plies": eval_res["avg_plies"],
                        "Avg Clock Used": f"{eval_res['avg_clock_used_s']}s",
                        "Mean ACPL": f"{eval_res['mean_acpl']} cp",
                    })
                st.dataframe(comp_rows, hide_index=True)
            except Exception as exc:
                st.error(f"Error running bullet evaluation: {exc}")
