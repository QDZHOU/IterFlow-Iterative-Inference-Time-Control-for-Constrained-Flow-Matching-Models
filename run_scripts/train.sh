#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HARDFLOW_NATIVE_MUJOCO=1
python -m run.train "$@"
