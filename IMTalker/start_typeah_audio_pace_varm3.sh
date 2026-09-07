#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VARIANT=AH
exec "$SCRIPT_DIR/start_winner_live.sh"
