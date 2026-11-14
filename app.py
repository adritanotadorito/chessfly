"""ChessFly: Interactive Fly-Wired Chess Policy Application.
Supports live play, human vs fly, bot vs bot, real 1+0 bullet matches,
transparent chess rewards, game replay, and reward-based RL training.
"""
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.svg
import numpy as np
import streamlit as st

from flychess.component import interactive_board
from flychess.games import list_games, load_game, save_game
from flychess.graph import load_graph
from flychess.match import (ChessMatch, format_clock, get_current_match,
                           set_current_match)
from flychess.model import FlyPolicy
from flychess.players import (FlyPolicyPlayer, GeminiPlayer, HumanPlayer,
                             StockfishPlayer, get_gemini_status)
from flychess.reward import compute_game_rewards
from flychess.train import (evaluate_bullet_matches, evaluate_heldout, generate,
                            read_examples, train_rl_bullet, train_supervised)

st.set_page_config(page_title="ChessFly", page_icon="🪰", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "data" / "flywire-subgraph.npz"
STOCKFISH_PATH = shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"

# Pre-RL and Post-RL checkpoint paths
FLY_SFT_CKPT = ROOT / "outputs" / "fly-200.npz"
FLY_RL_CKPT = ROOT / "outputs" / "fly-rl-bullet.npz"
SHUF_SFT_CKPT = ROOT / "outputs" / "shuffled-200.npz"
SHUF_RL_CKPT = ROOT / "outputs" / "shuffled-rl-bullet.npz"

# Fallback to 40-game checkpoints if 200-game not found
if not FLY_SFT_CKPT.is_file() and (ROOT / "outputs" / "fly.npz").is_file():
    FLY_SFT_CKPT = ROOT / "outputs" / "fly.npz"
if not SHUF_SFT_CKPT.is_file() and (ROOT / "outputs" / "shuffled.npz").is_file():
    SHUF_SFT_CKPT = ROOT / "outputs" / "shuffled.npz"


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def create_player(choice_key: str, custom_name: str = None):
    """Instantiate player object from UI selection."""
    if choice_key == "human":
        return HumanPlayer()
    elif choice_key == "fly_rl_pilot":
        ckpt = ROOT / "outputs" / "fly-rl-pilot.npz"
        return FlyPolicyPlayer(str(ckpt), str(GRAPH_PATH), name=custom_name or "Fly Policy (RL Pilot · Actor-Critic)")
    elif choice_key == "fly_sft_2k":
        ckpt = ROOT / "outputs" / "fly-supervised-2k.npz"
        return FlyPolicyPlayer(str(ckpt), str(GRAPH_PATH), name=custom_name or "Fly Policy (2k SFT · Oct-Validated)")
    elif choice_key == "fly_rl":
        ckpt = FLY_RL_CKPT if FLY_RL_CKPT.is_file() else FLY_SFT_CKPT
        return FlyPolicyPlayer(str(ckpt), str(GRAPH_PATH), name=custom_name or "Fly Policy (RL Bullet)")
    elif choice_key == "fly_sft":
        return FlyPolicyPlayer(str(FLY_SFT_CKPT), str(GRAPH_PATH), name=custom_name or "Fly Policy (Pre-RL SFT)")
    elif choice_key == "shuf_sft_2k":
        ckpt = ROOT / "outputs" / "shuffled-supervised-2k.npz"
        return FlyPolicyPlayer(str(ckpt), str(GRAPH_PATH), name=custom_name or "Shuffled Control (2k SFT)", shuffled=True)
    elif choice_key == "shuf_rl_pilot":
        ckpt = ROOT / "outputs" / "shuffled-rl-pilot.npz"
        return FlyPolicyPlayer(str(ckpt), str(GRAPH_PATH), name=custom_name or "Shuffled Control (RL Pilot)", shuffled=True)
    elif choice_key == "shuffled":
        ckpt = SHUF_RL_CKPT if SHUF_RL_CKPT.is_file() else SHUF_SFT_CKPT
        return FlyPolicyPlayer(str(ckpt), str(GRAPH_PATH), name=custom_name or "Shuffled Control", shuffled=True)
    elif choice_key == "stockfish_weak":
        return StockfishPlayer(STOCKFISH_PATH, difficulty="weak", name="Stockfish (Weak · Elo 1350)")
    elif choice_key == "stockfish_medium":
        return StockfishPlayer(STOCKFISH_PATH, difficulty="medium", name="Stockfish (Medium · Elo 1600)")
    elif choice_key == "stockfish_strong":
        return StockfishPlayer(STOCKFISH_PATH, difficulty="strong", name="Stockfish (Strong · Full)")
    elif choice_key == "gemini":
        return GeminiPlayer(name="Gemini (gemini-2.5-flash)")
    return HumanPlayer()


# Header Banner
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.title("🪰 ChessFly")
    st.caption("A fruit-fly connectome chess agent trained on real FlyWire synaptic circuits.")
with col_h2:
    gemini_status = get_gemini_status()
    st.markdown(
        f"<div style='text-align:right; padding-top:10px; font-size:13px; color:#94a3b8;'>"
        f"FlyWire: <span style='color:#4ade80;'>256 neurons · 396 edges</span><br>"
        f"Gemini API: {'<span style=\"color:#4ade80;\">Configured</span>' if gemini_status['is_configured'] else '<span style=\"color:#94a3b8;\">Not configured</span>'}"
        f"</div>",
        unsafe_allow_html=True
    )

# Primary Navigation Tabs
tab_play, tab_games, tab_rl, tab_sft, tab_about = st.tabs([
    "⚔️ Play & Live Matches",
    "📜 Games & Replay",
    "🧠 Reward-Based RL",
    "📊 SFT / Move Imitation",
    "🔬 Science & Architecture"
])

# ==============================================================================
# TAB 1: PLAY & LIVE MATCHES
# ==============================================================================
with tab_play:
    # Check if there is an active match
    active_match = get_current_match()

    # Setup bar if no game in progress or when resetting
    with st.expander("Match Setup & Player Selection", expanded=(active_match is None or active_match.game_over)):
        c_p1, c_p2, c_p3 = st.columns(3)
        player_options = {
            "human": "🧑 Human (You)",
            "fly_rl_pilot": "🪰 Fly Policy (RL Pilot · Actor-Critic)",
            "fly_sft_2k": "🪰 Fly Policy (2k SFT · Oct-Validated)",
            "fly_rl": "🪰 Fly Policy (Previous RL Bullet)",
            "fly_sft": "🪰 Fly Policy (Previous 200-SFT)",
            "shuf_sft_2k": "🔀 Shuffled Control (2k SFT)",
            "shuf_rl_pilot": "🔀 Shuffled Control (RL Pilot)",
            "shuffled": "🔀 Shuffled Control (Previous)",
            "stockfish_weak": "🤖 Stockfish (Weak · Elo 1350)",
            "stockfish_medium": "🤖 Stockfish (Medium · Elo 1600)",
            "stockfish_strong": "🤖 Stockfish (Strong · Skill 20)",
        }
        if gemini_status["is_configured"]:
            player_options["gemini"] = "✨ Gemini (gemini-2.5-flash)"
        else:
            player_options["gemini"] = "✨ Gemini (Disabled · Not configured)"

        with c_p1:
            p_white = st.selectbox("⚪ White Player", list(player_options.keys()), index=0,
                                   format_func=lambda k: player_options[k])
        with c_p2:
            p_black = st.selectbox("⚫ Black Player", list(player_options.keys()), index=1,
                                   format_func=lambda k: player_options[k])
        with c_p3:
            time_ctrl = st.selectbox("⏱️ Time Control", ["bullet", "analysis"],
                                     format_func=lambda k: "1+0 Bullet (60s, timed clocks)" if k == "bullet" else "Untimed Analysis")

        start_btn_col1, start_btn_col2, start_btn_col3 = st.columns([1, 1, 3])
        with start_btn_col1:
            if st.button("Start New Match", type="primary"):
                # Validate Gemini configuration if selected
                if (p_white == "gemini" or p_black == "gemini") and not gemini_status["is_configured"]:
                    st.error("Cannot start with Gemini: No API key configured. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or a .env file.")
                else:
                    if active_match is not None and not active_match.game_over:
                        active_match.game_over = True
                    white_obj = create_player(p_white)
                    black_obj = create_player(p_black)
                    new_match = ChessMatch(
                        white_player=white_obj,
                        black_player=black_obj,
                        timed=(time_ctrl == "bullet"),
                        base_time=60.0,
                    )
                    new_match.start_background()
                    set_current_match(new_match)
                    st.rerun()

        with start_btn_col2:
            if active_match is not None:
                if st.button("Reset Board"):
                    active_match.game_over = True
                    set_current_match(None)
                    st.rerun()

    # Board Display & Live Match Handling
    active_match = get_current_match()
    is_fresh = (active_match is None)

    if is_fresh:
        # Default fresh board state so board always renders on page load
        fresh_b = chess.Board()
        state = {
            "fen": fresh_b.fen(),
            "legal_moves": [m.uci() for m in fresh_b.legal_moves],
            "white_player": player_options.get(p_white, "White"),
            "black_player": player_options.get(p_black, "Black"),
            "white_clock": 60.0 if time_ctrl == "bullet" else None,
            "black_clock": 60.0 if time_ctrl == "bullet" else None,
            "formatted_white_clock": "1:00" if time_ctrl == "bullet" else "∞",
            "formatted_black_clock": "1:00" if time_ctrl == "bullet" else "∞",
            "turn": "white",
            "last_move": None,
            "fly_state": "idle",
            "moves": [],
            "ply_count": 0,
            "is_human_turn": (p_white == "human"),
            "is_game_over": False,
            "result": "*",
            "termination": "Ready to play",
            "orientation": "white",
        }
    else:
        state = active_match.get_state()

    # Board options bar (fallback toggle)
    c_ctrl1, c_ctrl2 = st.columns([4, 1])
    with c_ctrl2:
        use_fallback = st.checkbox("SVG Fallback Board", value=False,
                                   help="Check to render a pure server-side SVG board if iframe custom components are blocked.")

    if use_fallback:
        # Pure SVG Fallback Board
        st.caption("Rendering SVG Fallback Board (Pure Server-Side)")
        cur_board = chess.Board(state["fen"])
        last_m = chess.Move.from_uci(state["last_move"]) if state.get("last_move") else None
        board_svg = chess.svg.board(cur_board, size=480, lastmove=last_m)
        fb_c1, fb_c2 = st.columns([3, 2])
        with fb_c1:
            st.markdown(f'<div style="max-width:480px">{board_svg}</div>', unsafe_allow_html=True)
        with fb_c2:
            st.write(f"**Turn**: `{state['turn'].title()}`")
            st.write(f"**White**: `{state['white_player']}` · `{state['formatted_white_clock']}`")
            st.write(f"**Black**: `{state['black_player']}` · `{state['formatted_black_clock']}`")
            if state["is_human_turn"] and not state["is_game_over"]:
                fb_moves = [m.uci() for m in cur_board.legal_moves]
                if fb_moves:
                    selected_uci = st.selectbox("Select Legal Move", fb_moves)
                    if st.button("Submit Move", type="primary"):
                        if is_fresh:
                            white_obj = create_player(p_white)
                            black_obj = create_player(p_black)
                            new_match = ChessMatch(
                                white_player=white_obj,
                                black_player=black_obj,
                                timed=(time_ctrl == "bullet"),
                                base_time=60.0,
                            )
                            new_match.start_background()
                            set_current_match(new_match)
                            time.sleep(0.05)
                            new_match.submit_human_move(selected_uci)
                            st.rerun()
                        else:
                            active_match.submit_human_move(selected_uci)
                            time.sleep(0.05)
                            st.rerun()
    else:
        # Render custom interactive board
        component_props = {
            "fen": state["fen"],
            "legal_moves": state["legal_moves"],
            "white_player": state["white_player"],
            "black_player": state["black_player"],
            "white_clock": state["white_clock"],
            "black_clock": state["black_clock"],
            "formatted_white_clock": state["formatted_white_clock"],
            "formatted_black_clock": state["formatted_black_clock"],
            "turn": state["turn"],
            "last_move": state["last_move"],
            "fly_state": state["fly_state"],
            "moves": state["moves"],
            "ply": state["ply_count"],
            "is_human_turn": state["is_human_turn"],
            "is_game_over": state["is_game_over"],
            "result": state["result"],
            "termination": state["termination"],
            "orientation": state.get("orientation", "white"),
        }

        # Call custom component
        event = interactive_board(**component_props, key="live_interactive_board")

        # Handle user move submission from board
        if event and event.get("type") == "user_move":
            move_uci = event.get("uci")
            if state["is_human_turn"]:
                if is_fresh:
                    # Starting a game from initial board action
                    white_obj = create_player(p_white)
                    black_obj = create_player(p_black)
                    new_match = ChessMatch(
                        white_player=white_obj,
                        black_player=black_obj,
                        timed=(time_ctrl == "bullet"),
                        base_time=60.0,
                    )
                    new_match.start_background()
                    set_current_match(new_match)
                    time.sleep(0.05)
                    new_match.submit_human_move(move_uci)
                    st.rerun()
                else:
                    success = active_match.submit_human_move(move_uci)
                    if success:
                        time.sleep(0.05)
                        st.rerun()

    # Auto-refresh if game is in progress and not waiting for human
    if not is_fresh and not state["is_game_over"] and not state["is_human_turn"]:
        time.sleep(0.4)
        st.rerun()

    # If game just concluded, show summary banner
    if not is_fresh and state["is_game_over"]:
        st.success(f"🏁 Game Complete: {state['result']} ({state['termination']})")

# ==============================================================================
# TAB 2: GAMES & REPLAY
# ==============================================================================
with tab_games:
    st.subheader("📜 Persisted Games & Interactive Replay")
    st.caption("Every match is saved to disk with full PGN, FENs, per-move clocks, and transparent reward audits.")

    recent_games = list_games(limit=40)
    if not recent_games:
        st.info("No saved games found yet. Play a match in the 'Play' tab or run RL training to create game records.")
    else:
        game_select_col, replay_ctrl_col = st.columns([2, 3])
        with game_select_col:
            selected_game_id = st.selectbox(
                "Select Game to Replay",
                [g["id"] for g in recent_games],
                format_func=lambda gid: next(
                    f"{g['white']} vs {g['black']} · {g['result']} ({g['mode']}) [{g['timestamp'][:16].replace('T', ' ')}]"
                    for g in recent_games if g["id"] == gid
                )
            )

        loaded_game = load_game(selected_game_id)
        if loaded_game:
            moves = loaded_game.get("moves", [])
            total_plies = len(moves)

            # Reset or clamp replay_ply when switching game ID
            if "replay_game_id" not in st.session_state or st.session_state.replay_game_id != selected_game_id:
                st.session_state.replay_game_id = selected_game_id
                st.session_state.replay_ply = total_plies

            st.session_state.replay_ply = max(0, min(total_plies, st.session_state.get("replay_ply", total_plies)))

            # Replay navigation bar
            nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns(5)
            if nav_c1.button("⏮ Start", key=f"btn_start_{selected_game_id}"):
                st.session_state.replay_ply = 0
                st.rerun()
            if nav_c2.button("◀ Prev", key=f"btn_prev_{selected_game_id}"):
                st.session_state.replay_ply = max(0, st.session_state.replay_ply - 1)
                st.rerun()
            if nav_c3.button("Next ▶", key=f"btn_next_{selected_game_id}"):
                st.session_state.replay_ply = min(total_plies, st.session_state.replay_ply + 1)
                st.rerun()
            if nav_c4.button("⏭ End", key=f"btn_end_{selected_game_id}"):
                st.session_state.replay_ply = total_plies
                st.rerun()
            with nav_c5:
                st.download_button("💾 Download PGN", loaded_game.get("pgn", ""), file_name=f"{selected_game_id}.pgn", key=f"dl_{selected_game_id}")

            if total_plies > 0:
                cur_ply = st.slider("Step Through Plies", 0, total_plies, st.session_state.replay_ply, key=f"sld_{selected_game_id}")
            else:
                st.caption(f"ℹ️ Game terminated before move 1: {loaded_game.get('termination_reason', 'Empty game')}")
                cur_ply = 0
            st.session_state.replay_ply = cur_ply

            # Reconstruct board safely at selected ply
            try:
                r_board = chess.Board(loaded_game.get("initial_fen", chess.STARTING_FEN))
            except Exception:
                r_board = chess.Board()

            last_m_uci = None
            w_clock = loaded_game.get("white_clock")
            b_clock = loaded_game.get("black_clock")

            # Guaranteed safe iteration: Never index beyond len(moves)
            for i in range(min(cur_ply, len(moves))):
                m_rec = moves[i]
                if isinstance(m_rec, dict) and "uci" in m_rec:
                    try:
                        r_board.push(chess.Move.from_uci(m_rec["uci"]))
                        last_m_uci = m_rec["uci"]
                    except Exception:
                        pass
                    if m_rec.get("white_clock") is not None:
                        w_clock = m_rec["white_clock"]
                    if m_rec.get("black_clock") is not None:
                        b_clock = m_rec["black_clock"]

            r_col_left, r_col_right = st.columns([1, 1])
            with r_col_left:
                replay_props = {
                    "fen": r_board.fen(),
                    "legal_moves": [],
                    "white_player": loaded_game.get("white_player", "White"),
                    "black_player": loaded_game.get("black_player", "Black"),
                    "white_clock": w_clock,
                    "black_clock": b_clock,
                    "formatted_white_clock": format_clock(w_clock),
                    "formatted_black_clock": format_clock(b_clock),
                    "turn": "white" if r_board.turn == chess.WHITE else "black",
                    "last_move": last_m_uci,
                    "fly_state": "idle",
                    "moves": moves,
                    "ply": cur_ply,
                    "is_human_turn": False,
                    "is_game_over": (cur_ply == total_plies),
                    "result": loaded_game.get("result", "*"),
                    "termination": loaded_game.get("termination_reason", ""),
                }
                replay_event = interactive_board(**replay_props, key=f"replay_{selected_game_id}")
                if replay_event and replay_event.get("type") == "select_ply":
                    target_ply = replay_event.get("ply")
                    if target_ply is not None and target_ply != st.session_state.replay_ply:
                        st.session_state.replay_ply = max(0, min(total_plies, target_ply))
                        st.rerun()

            with r_col_right:
                st.subheader("Match Metadata & Diagnostics")
                st.write(f"- **Players**: `{loaded_game.get('white_player')}` vs `{loaded_game.get('black_player')}`")
                st.write(f"- **Result**: `{loaded_game.get('result')}` ({loaded_game.get('termination_reason')})")
                st.write(f"- **Mode**: `{loaded_game.get('mode')}` · Plies: `{total_plies}`")

                if cur_ply > 0 and cur_ply <= len(moves):
                    active_ply_info = moves[cur_ply - 1]
                    st.markdown(f"**Move {cur_ply}**: `{active_ply_info['player']}` played **`{active_ply_info['san']}`** (`{active_ply_info['uci']}`) in `{active_ply_info.get('elapsed_ms', 0)}ms`")

                # If reward diagnostics are available
                rewards_data = loaded_game.get("rewards")
                if rewards_data:
                    diag = rewards_data.get("diagnostics", {})
                    st.divider()
                    st.markdown("**Transparent Reward Audit**")
                    st.write(f"- **Final Match Outcome**: `{diag.get('terminal_reward', 0.0):+.1f}`")
                    st.write(f"- **Total Training Reward**: `{diag.get('total_training_reward', 0.0):+.4f}`")
                    st.write(f"- **Mean Centipawn Loss (ACPL)**: `{diag.get('mean_centipawn_loss', 0.0)} cp`")
                    st.write(f"- **Blunders (>300 cp)**: `{diag.get('blunder_count', 0)}`")
                    st.write(f"- **Bounded Stockfish Regret Penalty**: `{diag.get('engine_penalty', 0.0):.4f}`")

# ==============================================================================
# TAB 3: REWARD-BASED RL TRAINING
# ==============================================================================
with tab_rl:
    st.subheader("🧠 Online Reward-Based Reinforcement Learning")
    st.markdown("""
    Unlike supervised warm starts, this trains the policy **directly from complete bullet games**.
    The model plays against Stockfish, receives the transparent chess reward (material, safe activity, coordination, king safety, passed pawns, and bounded engine regret),
    and updates **the 396 biological FlyWire edge gains** as well as the dense adapter and move readout via policy gradients (**REINFORCE**).
    """)

    rl_c1, rl_c2, rl_c3 = st.columns(3)
    with rl_c1:
        rl_source_choice = st.selectbox("Warm Start Checkpoint", [
            ("outputs/fly-200.npz", False),
            ("outputs/shuffled-200.npz", True),
        ], format_func=lambda x: f"{Path(x[0]).name} {'(FlyWire)' if not x[1] else '(Shuffled Control)'}")
    with rl_c2:
        rl_match_count = st.number_input("Matches to Play", min_value=2, max_value=20, value=6, step=2)
    with rl_c3:
        rl_out_filename = st.text_input("Output RL Checkpoint", "outputs/fly-rl-bullet.npz")

    start_rl_btn = st.button("🚀 Run Online Bullet RL Training", type="primary")

    if start_rl_btn:
        out_p = ROOT / rl_out_filename
        ensure_parent(out_p)
        with st.spinner(f"Running RL training across {rl_match_count} bullet games against Stockfish…"):
            try:
                ckpt_p, is_shuf = rl_source_choice
                report = train_rl_bullet(
                    graph=str(GRAPH_PATH),
                    checkpoint=str(ROOT / ckpt_p),
                    output=str(out_p),
                    matches=int(rl_match_count),
                    shuffled=is_shuf,
                    stockfish_path=STOCKFISH_PATH,
                    bot_difficulty="weak",
                    lr=0.0005,
                )
                st.session_state["rl_train_report"] = report
                st.success(f"RL training complete! Saved updated checkpoint to `{rl_out_filename}`.")
            except Exception as e:
                st.error(f"RL training failed: {e}")

    if "rl_train_report" in st.session_state:
        rep = st.session_state["rl_train_report"]

        st.markdown("#### Completed RL Matches")
        st.dataframe(rep["history"], hide_index=True)

        st.markdown("#### Verified Weight Changes")
        st.caption("Demonstrating that the 396 biological FlyWire edge gains were actively updated by chess rewards.")
        diff_rows = []
        for k, v in rep["parameter_changes"].items():
            diff_rows.append({
                "Parameter Group": k,
                "L2 Norm Shift": v["l2_norm"],
                "Max Absolute Shift": v["max_abs"],
                "Mean Absolute Shift": v["mean_abs"],
            })
        st.dataframe(diff_rows, hide_index=True)

    st.divider()
    st.markdown("#### Matched Pre-RL vs. Post-RL Bullet Evaluation")
    st.caption("Evaluates fresh 1+0 bullet games alternating colors against Stockfish.")

    if st.button("Run 4-Game Bullet Evaluation"):
        with st.spinner("Running 4-game matched evaluations…"):
            try:
                eval_models = [
                    ("Pre-RL Fly (fly-200.npz)", str(FLY_SFT_CKPT), False),
                    ("Post-RL Fly (fly-rl-bullet.npz)", str(FLY_RL_CKPT), False) if FLY_RL_CKPT.is_file() else None,
                    ("Pre-RL Shuffled Control (shuffled-200.npz)", str(SHUF_SFT_CKPT), True),
                    ("Post-RL Shuffled Control (shuffled-rl-bullet.npz)", str(SHUF_RL_CKPT), True) if SHUF_RL_CKPT.is_file() else None,
                ]
                comp_rows = []
                for entry in eval_models:
                    if not entry or not Path(entry[1]).is_file():
                        continue
                    lbl, ckpt_p, is_shuf = entry
                    pol = FlyPolicy(str(GRAPH_PATH), shuffled=is_shuf)
                    pol.load_weights(ckpt_p)
                    res = evaluate_bullet_matches(pol, stockfish_path=STOCKFISH_PATH, num_games=4, bot_difficulty="weak")
                    comp_rows.append({
                        "Model": lbl,
                        "Record (W/D/L)": f"{res['wins']}/{res['draws']}/{res['losses']}",
                        "Win Rate": f"{res['win_rate']:.1%}",
                        "Flags Lost": res["flags_lost"],
                        "Avg Plies": res["avg_plies"],
                        "Avg Clock Used": f"{res['avg_clock_used_s']}s",
                        "Mean ACPL": f"{res['mean_acpl']} cp",
                    })
                st.dataframe(comp_rows, hide_index=True)
            except Exception as exc:
                st.error(f"Error during bullet evaluation: {exc}")

    st.divider()
    st.markdown("#### 🎯 Frozen November Benchmark Evaluation (120 Positions)")
    st.caption("Strictly held-out, frozen test set from November 2013 Lichess standard games (100 representative + 20 tactical challenge positions).")

    bench_rep_file = ROOT / "outputs" / "november_benchmark_evaluation_report.json"
    if bench_rep_file.is_file():
        try:
            with open(bench_rep_file) as f:
                b_report = json.load(f)
            b_rows = []
            for m_name, m_val in b_report.items():
                bm = m_val["benchmark"]
                tac = bm["tactical_challenge"]
                b_rows.append({
                    "Model": m_name,
                    "Legal Moves": f"{bm['legal_rate']*100:.1f}%",
                    "ACPL": f"{bm['mean_acpl']:.1f} cp",
                    "Median CPL": f"{bm['median_cpl']:.1f} cp",
                    "Blunders (>300cp)": f"{bm['blunder_rate']*100:.1f}%",
                    "Acceptable (<=50cp)": f"{bm['acceptable_move_rate']*100:.1f}%",
                    "Top-1 Accuracy": f"{bm['top1_accuracy']*100:.1f}%",
                    "Tactical Top-1": f"{tac['top1_accuracy']*100:.1f}%",
                    "Value MSE": f"{bm['value_mse']:.4f}",
                })
            st.dataframe(b_rows, hide_index=True)
        except Exception as e:
            st.error(f"Error loading benchmark report: {e}")

# ==============================================================================
# TAB 4: SFT / MOVE IMITATION
# ==============================================================================
with tab_sft:
    st.subheader("📊 Supervised Fine-Tuning (SFT / Move Imitation)")
    st.warning("⚠️ **Important Distinction**: Teacher move accuracy is supervised imitation on static positions, **NOT** bullet win rate or reinforcement learning.")

    sft_c1, sft_c2 = st.columns(2)
    with sft_c1:
        st.markdown("**1 · Generate Teacher Dataset**")
        n_games = st.number_input("Number of Games", min_value=10, max_value=1000, value=200, step=20)
        out_data_path = st.text_input("Dataset output path", "data/teacher-200.jsonl")
        if st.button("Generate Stockfish Positions"):
            try:
                dest = ROOT / out_data_path
                ensure_parent(dest)
                with st.spinner(f"Generating positions from {n_games} games…"):
                    gen_rep = generate(str(dest), STOCKFISH_PATH, games=int(n_games))
                st.success("Generated dataset!")
                st.json(gen_rep)
            except Exception as e:
                st.error(str(e))

    with sft_c2:
        st.markdown("**2 · Train SFT Policy & Control**")
        dataset_in = st.text_input("Dataset to train on", "data/teacher-200.jsonl")
        epochs = st.number_input("Epochs", min_value=1, max_value=20, value=8)
        prefix = st.text_input("Output Prefix", "outputs/sft-run")

        if st.button("Train Both Models (Best-Epoch Checkpointing)"):
            try:
                f_ckpt = ROOT / f"{prefix}-fly.npz"
                s_ckpt = ROOT / f"{prefix}-shuffled.npz"
                ensure_parent(f_ckpt)
                with st.spinner("Training real flywire and shuffled models…"):
                    f_rep = train_supervised(str(GRAPH_PATH), str(ROOT / dataset_in), str(f_ckpt), epochs=int(epochs))
                    s_rep = train_supervised(str(GRAPH_PATH), str(ROOT / dataset_in), str(s_ckpt), epochs=int(epochs), shuffled=True)
                st.success("SFT Complete!")
                st.json({
                    "Fly Best Top-1": f_rep["best_heldout"]["top1"],
                    "Fly Best Loss": f_rep["best_heldout"]["loss"],
                    "Shuffled Best Top-1": s_rep["best_heldout"]["top1"],
                    "Shuffled Best Loss": s_rep["best_heldout"]["loss"],
                })
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown("#### Held-Out Move Prediction Metrics")
    teacher_ds = ROOT / "data" / "teacher-200.jsonl"
    if teacher_ds.is_file():
        try:
            _, examples = read_examples(str(teacher_ds))
            held = [r for r in examples if r["split"] == "heldout"]
            st.write(f"Evaluating across **{len(held)} held-out positions**:")
            sft_eval_rows = []
            for lbl, p_ckpt, is_shuf in [
                ("FlyWire · Untrained", None, False),
                ("FlyWire · SFT (fly-200.npz)", FLY_SFT_CKPT, False),
                ("Shuffled Control · Untrained", None, True),
                ("Shuffled Control · SFT (shuffled-200.npz)", SHUF_SFT_CKPT, True),
            ]:
                if p_ckpt is not None and not Path(p_ckpt).is_file():
                    continue
                pol = FlyPolicy(str(GRAPH_PATH), shuffled=is_shuf)
                if p_ckpt:
                    pol.load_weights(str(p_ckpt))
                m = evaluate_heldout(pol, held)
                sft_eval_rows.append({
                    "Model": lbl,
                    "Top-1 Match": f"{m['top1']:.1%}",
                    "Top-3 Match": f"{m['top3']:.1%}",
                    "Top-5 Match": f"{m['top5']:.1%}",
                    "Validation Loss": m["loss"],
                })
            st.dataframe(sft_eval_rows, hide_index=True)
        except Exception as exc:
            st.error(f"Error loading SFT evaluation: {exc}")

# ==============================================================================
# TAB 5: SCIENCE & ARCHITECTURE
# ==============================================================================
with tab_about:
    st.subheader("🔬 Connectome Constraints & Scientific Methodology")
    st.markdown(r"""
    ### Biological Grounding
    - **Connectome Source**: Adult *Drosophila melanogaster* whole-brain connectome (FlyWire FAFB v783 release).
    - **Extracted Subgraph**: 256 neurons, 396 synaptic connections extracted via highest-synapse connected expansion.
    - **Recurrent Dynamics**: Every recurrent message-passing edge corresponds strictly to an authentic biological synapse. The model optimizes one biological gain parameter $g_e \in [-2, 2]$ per connection.
    - **Engineered Periphery**: A dense linear adapter projects 785 board and clock features into the 256 neurons; a linear readout maps neuron activations to legal move candidates.

    ### Why the Shuffled Control Matters
    To test whether the specific biological wiring diagram confers an inductive bias for chess representation, we construct a degree-preserving randomized graph with identical in-degree and out-degree distributions and normalized synaptic weight scales.

    ### Transparent Reward-Based Formulation
    - **6 Configurable Sub-Scores (Sum = 1.0)**:
      1. *Material Balance* ($w = 0.28$): $\tanh(\Delta M / 4.0) \in (-1, 1)$.
      2. *Safe Piece Activity* ($w = 0.12$): Mobility excluding squares attacked by enemy pawns or lesser pieces.
      3. *Coordination & Defended Pieces* ($w = 0.08$): Fraction of friendly pieces defended by other pieces.
      4. *Opponent Threats & Counterplay* ($w = 0.20$): Rigorous exchange evaluation where capturing defended pieces with higher-value pieces registers as negative, winning undefended pieces registers positively, and equal trades register neutrally.
      5. *Pawn Structure* ($w = 0.12$): Passed pawn advancement bonus minus doubled and isolated pawn penalties.
      6. *King Safety & Centralization* ($w = 0.20$): Game-phase blended ($\rho = \text{npm}/32$) between middlegame pawn shelter/open-file danger and endgame king centralization.
    - **State-Dependent Potential**: $\Phi(s, c) = \sum_{k} w_k \hat{f}_k(s, c) \in (-1, 1)$.
    - **Intermediate Transition Reward**: Evaluated after opponent reply: $\Delta \Phi_t = 0.15 \times (0.99 \Phi(s_{t+1}) - \Phi(s_t))$.
    - **Bounded Regret Penalty**: $0.0$ for loss $\le 20$ cp, up to $-0.05$ for small inaccuracy, $-0.05$ to $-0.20$ for mistakes, and $-0.20$ to $-0.50$ for tactical blunders.
    - **Terminal Dominance**: $R_{\text{terminal}} \in \{+1.0, 0.0, -1.0\}$.

    ### Advantage Actor-Critic (A2C)
    A value head $V(s) = \tanh(W_v x + b_v)$ on the 256 biological neurons computes state value estimates. Advantage $A_t = G_t - V(s_t)$ scales policy updates while value MSE backpropagates through biological recurrence, directly tuning the 396 FlyWire edge gains $g_e$.

    ### Strict Split Methodology
    - **Training Split**: January–September 2013 Lichess standard games (filtered $\ge 1500$ Elo, normal termination, FEN deduplication).
    - **Validation Split**: October 2013 games for checkpoint selection and early stopping.
    - **Frozen Benchmark**: November 2013 games (100 representative + 20 tactical challenge positions) strictly frozen for final evaluation.

    """)

    if GRAPH_PATH.is_file():
        try:
            _, _, _, meta_info, _ = load_graph(str(GRAPH_PATH))
            st.json(meta_info)
        except Exception:
            pass
