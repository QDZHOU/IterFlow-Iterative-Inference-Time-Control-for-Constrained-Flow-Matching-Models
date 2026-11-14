#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m run.eval run --method hardflow "$@"
