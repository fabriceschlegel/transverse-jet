#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$TASK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
exec "$TASK_ROOT/.venv/bin/python" -m jetflow.cli "$@"
