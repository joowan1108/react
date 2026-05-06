#!/bin/sh
set -eu

mkdir -p /logs /output /tmp/data-agent-submission

uv run python /app/main.py 2>&1 | tee /logs/runtime.log
