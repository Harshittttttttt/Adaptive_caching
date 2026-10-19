@echo off
python -m pip install -r requirements.txt
set PYTHONPATH=src
python scripts\run_benchmark.py --dataset synthetic --seeds 3 --queries 20 --output results\synthetic_quick
pause
