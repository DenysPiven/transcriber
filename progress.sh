#!/bin/bash
# Live progress bar for a running transcribe.py job.
# Usage: ./progress.sh [log_file]
LOG="${1:-$(ls -t "$(dirname "$0")"/*.log | head -1)}"
WIDTH=40

while true; do
  # Progress lines are separated by \r; take the last one.
  last=$(tr '\r' '\n' < "$LOG" | grep -E '^\[[0-9:]+ / [0-9:]+\]' | tail -1)
  # The final line ends with the .srt path.
  done_line=$(grep -E '\.srt$' "$LOG")
  if [[ -n "$done_line" ]]; then
    printf "\r%-80s\n" "Done: ${done_line#* }"
    exit 0
  fi
  if grep -qE "Traceback|Killed" "$LOG"; then
    printf "\nFailed, see %s\n" "$LOG"; exit 1
  fi
  if [[ -n "$last" ]]; then
    pos=$(sed -E 's/^\[([0-9:]+) \/.*/\1/' <<<"$last")
    total=$(sed -E 's/^\[[0-9:]+ \/ ([0-9:]+)\].*/\1/' <<<"$last")
    pct=$(sed -E 's/.*\] *([0-9.]+)%.*/\1/' <<<"$last")
    eta=$(sed -E 's/.*ETA ([0-9:]+).*/\1/' <<<"$last")
    filled=$(awk -v p="$pct" -v w=$WIDTH 'BEGIN{printf "%d", p*w/100}')
    bar=$(printf "%${filled}s" "" | tr ' ' '#')$(printf "%$((WIDTH-filled))s" "" | tr ' ' '-')
    printf "\r[%s] %5.1f%%  %s / %s  ETA %s  " "$bar" "$pct" "$pos" "$total" "$eta"
  else
    printf "\rWaiting for first segments..."
  fi
  sleep 2
done
