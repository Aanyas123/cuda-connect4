#!/usr/bin/env bash
# Builds the game and plays three GPU-vs-GPU matches, saving logs, move
# lists, and board replays to artifacts/.
# Usage: ./run.sh [ARCH] [MINIMAX_DEPTH]
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${1:-sm_86}"
DEPTH="${2:-7}"
DELAY_MS="${DELAY_MS:-150}"  # Pause after each move so the game is watchable.

make ARCH="${ARCH}"
mkdir -p artifacts games

# Give each competitor its own GPU when more than one is available.
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
P2_DEVICE=0
if [ "${NUM_GPUS:-0}" -ge 2 ]; then P2_DEVICE=1; fi
{ date -u; nvidia-smi; } > artifacts/gpu_info.txt 2>&1 || true
: > artifacts/summary.txt

# play_game NAME P1_STRATEGY P2_STRATEGY
play_game() {
  local name="$1" s1="$2" s2="$3" dir="games/$1"
  rm -rf "${dir}"
  ./bin/connect4 --init --game_dir "${dir}"
  ./bin/connect4 --play 1 --strategy "${s1}" --game_dir "${dir}" \
      --depth "${DEPTH}" --seed 1 --device 0 --move_delay_ms "${DELAY_MS}" \
      > "artifacts/${name}_p1.log" 2>&1 &
  local pid1=$!
  ./bin/connect4 --play 2 --strategy "${s2}" --game_dir "${dir}" \
      --depth "${DEPTH}" --seed 2 --device "${P2_DEVICE}" \
      --move_delay_ms "${DELAY_MS}" > "artifacts/${name}_p2.log" 2>&1 &
  local pid2=$!
  ./bin/connect4 --watch --game_dir "${dir}" --log "artifacts/${name}_board.txt"
  wait "${pid1}" "${pid2}"
  cp "${dir}/moves.csv" "artifacts/${name}_moves.csv"
  echo "${name}: P1=${s1} vs P2=${s2} -> $(tail -n 1 "artifacts/${name}_board.txt")" \
      | tee -a artifacts/summary.txt
}

play_game game1 mc minimax
play_game game2 minimax mc
play_game game3 mc mc

echo "=== Summary ==="
cat artifacts/summary.txt

if ! python3 -c "import PIL" 2>/dev/null; then
  pip3 install --user pillow || true
fi
for g in game1 game2 game3; do
  python3 scripts/render_game.py "artifacts/${g}_moves.csv" "artifacts/${g}" || true
done
