#!/bin/bash
set -e
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
python scripts/build_index.py
python scripts/run_experiments.py --limit 3
