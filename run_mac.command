#!/usr/bin/env bash
# ============================================================
#  ALMA - start the app on macOS
#  Double-click this after setup_mac.command has finished.
#  First time only: right-click > Open, then confirm.
# ============================================================
cd "$(dirname "$0")"
bash ./run.sh "$@"
echo
echo "The app has stopped."
read -r -p "Press Enter to close this window. "
