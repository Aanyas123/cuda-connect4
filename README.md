# GPU vs. GPU Connect Four

Two CUDA programs compete at Connect Four. Each competitor is its own
process with its own CUDA context (and its own GPU when more than one is
present). The two processes take turns through a shared game directory
protected by a lock file, and a third process draws the board after every
move.

| Competitor | Strategy | GPU work per move |
|---|---|---|
| `mc` | Monte Carlo: random playouts for every candidate column | 7 x 64 x 256 threads x 4 playouts = ~459k games |
| `minimax` | Parallel minimax: one thread per 7-move sequence + level-by-level reduction | 7^7 = 823,543 leaves |

## Requirements
- NVIDIA GPU, CUDA Toolkit (`nvcc`), GNU make, Linux (uses `flock`)
- Python 3 + Pillow (optional, only for GIF replays)

## Quick start
```bash
./run.sh sm_86 7        # args: GPU arch (default sm_86), minimax depth (default 7)
```
`run.sh` builds the program and plays three matches (mc vs. minimax,
minimax vs. mc, mc vs. mc). For each match it saves these files to `artifacts/`:
- `gameN_board.txt`: every board position, plus the result
- `gameN_moves.csv`: move, player, strategy, column, score, GPU ms, work
- `gameN_p1.log` and `gameN_p2.log`: each GPU's reasoning, with all 7 column scores per move
- `gameN.gif` and `gameN_final.png`: an animated replay

It also writes `summary.txt` (the results) and `gpu_info.txt` (the `nvidia-smi` output).
Set `DELAY_MS=0 ./run.sh` to play at full speed.

## Running the pieces by hand
```bash
make ARCH=sm_86
./bin/connect4 --init --game_dir games/demo
./bin/connect4 --play 1 --strategy mc      --game_dir games/demo --move_delay_ms 300 &
./bin/connect4 --play 2 --strategy minimax --game_dir games/demo --depth 7 &
./bin/connect4 --watch --game_dir games/demo
```

| Flag | Meaning |
|---|---|
| `--init` / `--play N` / `--watch` | Create a game / join as player N (1 moves first) / display it |
| `--game_dir DIR` | Shared game directory (required) |
| `--strategy mc\|minimax` | Competitor's algorithm |
| `--depth D` | Minimax depth 1-8 (default 7) |
| `--blocks N`, `--playouts N` | Monte Carlo blocks per column (64) and playouts per thread (4) |
| `--seed S` | Monte Carlo seed |
| `--device N` | GPU used by this player |
| `--move_delay_ms N` | Pause after each move so humans can follow |
| `--timeout_s N` | Abort if the game stalls (default 120) |
| `--log FILE` | (watch) also save plain-text boards |

## Design

### Taking turns: two processes, one lock file
`games/<name>/state.txt` holds `turn`, `status`, the move list, and both
strategy names. Every read or write happens while holding an exclusive
`flock()` on `games/<name>/lock`. Writes go to `state.txt.tmp` and are then
`rename()`d over the real file, so a reader can never see half a file. Each
player runs this loop:
1. Take the lock, read the state, release the lock. If the game is over, exit.
   If it's not this player's turn, sleep 5 ms and check again.
2. Rebuild the board from the move list and run its GPU strategy. No lock is
   held while the GPU is thinking.
3. Take the lock again and confirm the state didn't change. Append the move,
   set the win/draw status, hand the turn to the opponent, and add a line to
   `moves.csv`.

The Coursera lab has only one GPU, so both processes share it, each with its
own CUDA context. With `--device`, each player can use a separate GPU, and
`run.sh` does this automatically when `nvidia-smi` reports two or more GPUs.

### Board representation (`src/board.h`)
The board is stored as a 64-bit bitboard (7 bits per column), shared by host
and device code through `__host__ __device__` functions. A move is two
operations: `current ^= mask; mask |= mask + bottom(col)`. Checking for four
in a row takes four shift-and-AND tests, one per direction. This keeps the
state per thread down to 20 bytes held in registers.

### Competitor 1: Monte Carlo (`MonteCarloKernel`)
- Grid `(64, 7)`: `blockIdx.y` is the candidate column, so each column gets
  64 blocks of 256 threads.
- Each thread plays 4 games to the end from the position after that column.
  It uses its own xorshift random generator, seeded with SplitMix64 from
  (seed, column, block, thread). Moves are random, except that a player who
  can win immediately always does. This keeps playouts realistic without
  slowing them down.
- Results are scored win = 2, draw = 1, loss = 0. They're reduced with warp
  `__shfl_down_sync`, then combined per warp in shared memory, then one
  64-bit `atomicAdd` per block goes to `totals[col]`.
- The host picks the column with the highest average score. Ties go to the
  most central column.

### Competitor 2: parallel minimax (`MinimaxLeafKernel` + `MinimaxReduceKernel`)
- The search tree is laid out as an implicit array. Leaf index `i`, written
  in base 7, is the move sequence, so the children of node `i` are
  `7i .. 7i+6`.
- **Leaf kernel:** thread `i` decodes its digits and plays the moves. It stops
  early with `+-(100000 - ply)` on a win or loss (this prefers faster wins),
  `0` on a draw, or `INT_MIN` if a move is illegal. Otherwise it scores the
  position with a heuristic: all 69 lines of four are stored in `__constant__`
  memory, each is counted with `__popcll`, open 2s and 3s are rewarded, and
  there's a bonus for the center column.
- **Reduction kernels:** `depth - 1` launches collapse one tree level each,
  taking the max on the mover's plies and the min on the opponent's, and
  skipping illegal children. What's left is 7 scores for the root, copied to
  the host.
- At depth 7, that's 823,543 threads for the leaves, plus reduction passes of
  117,649, 16,807, 2,401, 343, 49 and 7 threads.

### Showing the game (`--watch`, `scripts/render_game.py`)
The watcher polls the state file and prints every new position as a colored
ASCII board: red X for player 1, yellow O for player 2, with a `v` marking
the last move. It also saves a plain-text copy. Each player's log prints the
scores it computed for all seven columns, so the reason for every move can be
seen. `render_game.py` turns `moves.csv` into an animated GIF.

## Challenges and lessons learned
- **Mapping a tree onto a GPU.** Minimax is naturally recursive, which GPUs
  handle poorly. Enumerating every sequence of moves as a flat array turns it
  into one embarrassingly parallel kernel plus a few small reductions. The
  cost is exhaustive search with no alpha-beta pruning, so depth 7 is cheap
  but depth 9 (40M leaves) would not be.
- **Illegal sequences.** Many base-7 sequences run into a full column. Using
  an `INT_MIN` sentinel that reductions skip keeps every thread doing the same
  work, with no compaction pass.
- **Playout quality vs. quantity.** In purely random playouts, a player
  often ignores a win that's available immediately, which distorts the
  statistics. The rule "always take a winning move" costs up to seven extra
  bit tests per ply and makes each playout far more realistic.
- **Coordination without shared memory between processes.** Atomic
  `rename()` plus `flock()` was enough to make the two separate programs take
  turns safely. Checking the state again after the GPU move catches any race.

## Repository layout
```
src/board.h            Bitboard (host + device)
src/strategies.cu/.cuh Monte Carlo and parallel minimax kernels
src/game_file.cc/.h    Lock file + atomic state file used for turn-taking
src/board_util.cc/.h   Rebuild board from moves, ASCII rendering
src/main.cu            CLI: --init / --play / --watch
scripts/render_game.py GIF replay from moves.csv
run.sh, Makefile       Build + three demo matches
PRESENTATION.md        Outline/script for the video presentation
```
