# Add the ChessFly dashboard to your existing repo

Copy `app.py`, `requirements-gui.txt`, and `start-gui.command` into the root of your existing `chessfly` folder, beside `flychess/`, `README.md`, `data/`, and `outputs/`. Do not replace your training outputs or source files.

Once, from the `chessfly` folder with `.venv` active:

```bash
pip install -r requirements-gui.txt
```

To open the dashboard later:

```bash
bash start-gui.command
```

Streamlit opens a local browser page (usually `http://localhost:8501`). The page explains the experiment, shows the real graph's provenance, displays the chessboard and model moves, compares existing checkpoints, and lets you generate a new dataset and train matched fly/shuffled models using buttons. It does not upload your data. Its work is local to your Mac.

The dashboard does **not** implement AfterQuery's judging harness, give a model chess strength it has not earned, or claim that a full fly brain plays chess. Your existing 40-game results are visible when the default paths match. Larger runs are saved under new names so the initial outputs remain intact.
