// GPU move-selection strategies for the two competitors.
#ifndef CONNECT4_STRATEGIES_CUH_
#define CONNECT4_STRATEGIES_CUH_

#include <cstdint>

#include <cuda_runtime.h>

#include "board.h"

namespace c4 {

constexpr int kMaxDepth = 8;  // 7^8 = 5.7M leaves.

struct MoveResult {
  int column = -1;
  double score = 0.0;               // Score of the chosen column.
  double column_scores[kWidth] = {};
  bool legal[kWidth] = {};
  int64_t work = 0;                 // Playouts or leaves evaluated.
  const char* work_unit = "";
  float gpu_ms = 0.0f;
};

struct MonteCarloConfig {
  int blocks_per_move = 64;
  int playouts_per_thread = 4;
  uint64_t seed = 1;
};

// Uploads constant tables. Call once per process after cudaSetDevice().
cudaError_t InitStrategies();

// Strategy "mc": for every legal column, runs blocks_per_move * 256 *
// playouts_per_thread random games on the GPU and picks the column with the
// highest average result (win = 1, draw = 0.5, loss = 0).
MoveResult MonteCarloMove(const Board& board, const MonteCarloConfig& config);

// Strategy "minimax": one GPU thread per move sequence of length |depth|
// (7^depth threads) scores its leaf with a heuristic, then |depth| - 1
// reduction kernels apply max/min level by level back to the root.
MoveResult MinimaxMove(const Board& board, int depth);

}  // namespace c4

#endif  // CONNECT4_STRATEGIES_CUH_
