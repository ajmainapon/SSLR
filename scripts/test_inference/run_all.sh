#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 preflight.py
python3 eval_test.py --workers 4
