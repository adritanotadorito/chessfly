"""Public scenario definitions for ChessFly: Counterplay.

Contains ONLY public FENs, side to move, number of turns, and scenario descriptions.
Does NOT contain reference solutions, engine evaluations, or secret labels.
"""

SCENARIOS = [
    {
        "id": "scenario_1_poisoned_material",
        "name": "Scenario 1: Poisoned Material",
        "side": "white",
        "fen": "rnb1kbnr/pp2pppp/2p5/8/2QP4/2N5/Pq2PPPP/R1B1KBNR w KQkq - 0 6",
        "agent_turns": 2,
        "max_inspections": 3,
        "description": "Black has greedily played Qxb2, attacking your rook on a1. Find the accurate refutation that traps or severely harasses the overextended Queen.",
    },
    {
        "id": "scenario_2_acute_threat_defense",
        "name": "Scenario 2: Acute Threat Defense",
        "side": "black",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
        "agent_turns": 2,
        "max_inspections": 3,
        "description": "White has launched a sharp Scholar's attack with Qh5 and Bc4, threatening immediate checkmate on f7. Find the exact parry that neutralizes the attack without dropping material.",
    },
    {
        "id": "scenario_3_tactical_counterplay",
        "name": "Scenario 3: Tactical Counterplay",
        "side": "black",
        "fen": "r1b1kbnr/pppp1Npp/8/6q1/2BnP3/8/PPPP1PPP/RNBQK2R b KQkq - 0 5",
        "agent_turns": 2,
        "max_inspections": 3,
        "description": "White has greedily played 5. Nxf7?, forking your Queen and Rook. Passive defense leaves you ruined. Find the sharp tactical counter-strike that turns the table and traps White.",
    },
    {
        "id": "scenario_4_sound_vs_unsound",
        "name": "Scenario 4: Sound vs. Unsound Decision",
        "side": "white",
        "fen": "r1bq1rk1/pp1n1ppp/2n1p3/2bpP3/3p1B2/2PB1N2/PP2NPPP/R2Q1RK1 w - - 0 10",
        "agent_turns": 2,
        "max_inspections": 3,
        "description": "A sharp French middlegame with central tension. A flashy bishop sacrifice on h7 is tempting but premature. Choose the sound central continuation that maintains a dominant position.",
    },
]
