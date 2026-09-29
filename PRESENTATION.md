# Video presentation script (about 6-8 minutes)

This follows the three grading categories in order. Record your screen: the
lab terminal plus this repo open in the editor.

## 1. Project description (about 1 min), graded "Project Description"
- "My project is Connect Four played by two GPU competitors."
- "Each competitor is a separate program with its own CUDA context. With two
  GPUs each gets its own device; the Coursera lab has one, so they share it."
- "They take turns through a shared game folder: a lock file and a state file
  that is replaced atomically."
- "The two competitors use different strategies: player `mc` uses Monte Carlo
  playouts, and player `minimax` uses a parallel minimax search."
- Show the README table.

## 2. Code description (about 3-4 min), graded "Code Description"
Walk through the turn of one competitor.

1. **Input:** in `main.cu` `RunPlayer`, show the loop: lock, read
   `state.txt`, and wait for `turn == me`. Then `BuildBoard` rebuilds the
   bitboard from the move list.
2. **Board:** in `board.h`, explain the two 64-bit masks, `Play` as two
   operations, and `HasFour` using shifts.
3. **Monte Carlo decision** (`strategies.cu`, `MonteCarloKernel`):
   - Grid (64, 7): blockIdx.y is the candidate column.
   - Each thread plays 4 random games from that position, always taking an
     immediate win (show `Playout`).
   - Results are reduced with warp shuffles, then shared memory, then an
     atomicAdd. The host picks the highest win rate.
4. **Minimax decision** (`MinimaxLeafKernel`, `MinimaxReduceKernel`):
   - Thread i reads i in base 7 as a 7-move sequence, which is 823,543
     threads.
   - Each leaf gets a win/loss score, or a heuristic based on the 69 lines of
     four in constant memory counted with `__popcll`. Illegal sequences get
     the INT_MIN sentinel.
   - 6 reduction launches alternate max and min back up to the 7 root scores.
5. **Output:** back in `RunPlayer`, the move is committed under the lock
   after re-checking the state. The turn passes to the opponent and a line is
   added to `moves.csv`.
6. Mention a challenge from the README's "Challenges and lessons learned"
   section, for example the sentinel for illegal moves, or having no
   alpha-beta pruning on the GPU.

## 3. Demonstration and visualization (about 2 min), graded "Demonstration/Visualization"
- In the lab terminal, run `./run.sh sm_86 7` and let game 1 play live. The
  colored board redraws after every move, with a `v` over the last column.
- Open `artifacts/game1_p1.log` and `game1_p2.log` to show each GPU's seven
  column scores, its chosen move, and the GPU time for every turn.
- Show `artifacts/summary.txt` with the three results, and play
  `artifacts/game1.gif`.
- Close with one sentence about which strategy won and why you think it did.
  Take this from your actual results.

## Recording tips
- Windows: Win + Alt + R (Xbox Game Bar) records the current window as MP4.
  OBS Studio or a Zoom meeting with "Record to this computer" also work.
- Zoom the lab terminal (Ctrl + =) so the board is readable.
- Upload the MP4 to the Coursera assignment. If it's too large, upload it to
  YouTube or Google Drive as unlisted and submit the link.
