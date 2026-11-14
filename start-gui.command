#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -f .venv/bin/activate ]; then
  echo "No .venv here. From this folder run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -r requirements-gui.txt"
  exit 1
fi
source .venv/bin/activate
python -m streamlit run app.py
