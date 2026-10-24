#!/usr/bin/env bash
# ============================================================
#  ALMA - one-time macOS setup
#  Double-click this file once. When it finishes, use
#  run_mac.command to start the app.
#  First time only: right-click > Open, then confirm.
# ============================================================
cd "$(dirname "$0")"
bash ./setup.sh "$@"
echo
read -r -p "Press Enter to close this window. "
