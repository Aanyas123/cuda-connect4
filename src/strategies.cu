#include "strategies.cuh"

#include <climits>
#include <cmath>
#include <utility>
#include <vector>

#include "cuda_check.cuh"

namespace c4 {
namespace {

constexpr int kThreads = 256;
constexpr int kNumWindows = 69;  // Every line of four cells on a 7x6 board.
constexpr int kWinScore = 100000;
constexpr int kInvalid = INT_MIN;  // Marks an illegal move sequence.
// Columns in tie-break order: central columns are stronger.
constexpr int kColumnOrder[kWidth] = {3, 2, 4, 1, 5, 0, 6};

__constant__ uint64_t kWindows[kNumWindows];

// ---------------------------------------------------------------------------
// Shared device helpers.
// ---------------------------------------------------------------------------

// Heuristic from the point of view of |own|: rewards open lines of 2 and 3
// stones and central stones, penalizes the same for the opponent.
__device__ int Evaluate(uint64_t own, uint64_t opp) {
  int score = 0;
  for (int i = 0; i < kNumWindows; ++i) {
    const uint64_t w = kWindows[i];  // Same i in every thread: broadcast read.
    const int mine = __popcll(own & w);
    const int theirs = __popcll(opp & w);
    if (theirs == 0) {
      score += (mine == 3) ? 5 : (mine == 2) ? 2 : 0;
    } else if (mine == 0) {
      score -= (theirs == 3) ? 5 : (theirs == 2) ? 2 : 0;
    }
  }
  const uint64_t center = 0x3FULL << (3 * (kHeight + 1));
  score += 3 * (__popcll(own & center) - __popcll(opp & center));
  return score;
}

__device__ uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// xorshift64* generator; one per thread, kept in registers.
struct Rng {
  uint64_t state;
  __device__ uint32_t Next() {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return static_cast<uint32_t>((state * 0x2545F4914F6CDD1DULL) >> 32);
  }
};

// ---------------------------------------------------------------------------
// Monte Carlo.
// ---------------------------------------------------------------------------

// Plays random moves from |b| (non-terminal) to the end of the game. A player
// who can win immediately always does, which makes playouts far more
// realistic than pure random play. Returns 2 / 1 / 0 for a win / draw / loss
// of |root_player|.
__device__ unsigned int Playout(Board b, int root_player, Rng* rng) {
  while (true) {
    int legal[kWidth];
    int num_legal = 0;
    for (int col = 0; col < kWidth; ++col) {
      if (!CanPlay(b, col)) continue;
      if (IsWinningMove(b, col)) {
        return PlayerToMove(b) == root_player ? 2u : 0u;
      }
      legal[num_legal++] = col;
    }
    // A random move is never a winning one (checked above), so only a full
    // board can end the game here.
    Play(&b, legal[rng->Next() % num_legal]);
    if (IsFull(b)) return 1u;
  }
}

// Grid: (blocks_per_move, kWidth). blockIdx.y is the candidate column.
__global__ void MonteCarloKernel(Board root, int playouts_per_thread,
                                 uint64_t seed, unsigned long long* totals) {
  const int col = blockIdx.y;
  if (!CanPlay(root, col)) return;  // Uniform per block, so no sync hazard.

  const int root_player = PlayerToMove(root);
  Board b = root;
  Play(&b, col);

  unsigned int local = 0;
  if (HasFour(LastMoverStones(b))) {
    local = 2u * playouts_per_thread;  // Immediate win.
  } else if (IsFull(b)) {
    local = 1u * playouts_per_thread;  // Immediate draw.
  } else {
    const uint64_t id = (static_cast<uint64_t>(col) << 40) |
                        (static_cast<uint64_t>(blockIdx.x) << 16) |
                        threadIdx.x;
    Rng rng{SplitMix64(seed ^ SplitMix64(id))};
    if (rng.state == 0) rng.state = 1;
    for (int i = 0; i < playouts_per_thread; ++i) {
      local += Playout(b, root_player, &rng);
    }
  }

  // Block reduction: warp shuffles, then one partial sum per warp.
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xffffffffu, local, offset);
  }
  __shared__ unsigned int warp_sums[kThreads / 32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  if (lane == 0) warp_sums[warp] = local;
  __syncthreads();
  if (warp == 0) {
    unsigned int v = (lane < kThreads / 32) ? warp_sums[lane] : 0u;
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    if (lane == 0) atomicAdd(&totals[col], static_cast<unsigned long long>(v));
  }
}

// ---------------------------------------------------------------------------
// Parallel minimax.
// ---------------------------------------------------------------------------

// Thread |idx| decodes its base-7 digits into a move sequence (first move =
// most significant digit, so the children of node i are 7i .. 7i + 6), plays
// it, and scores the result from the root player's point of view.
__global__ void MinimaxLeafKernel(Board root, int depth, int num_leaves,
                                  int* scores) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_leaves) return;

  int cols[kMaxDepth];
  int rest = idx;
  for (int k = depth - 1; k >= 0; --k) {
    cols[k] = rest % kWidth;
    rest /= kWidth;
  }

  const int root_player = PlayerToMove(root);
  Board b = root;
  for (int k = 0; k < depth; ++k) {
    if (!CanPlay(b, cols[k])) {
      scores[idx] = kInvalid;
      return;
    }
    Play(&b, cols[k]);
    if (HasFour(LastMoverStones(b))) {
      // Faster wins (and slower losses) score better. Even plies are root's.
      const int s = kWinScore - (k + 1);
      scores[idx] = (k % 2 == 0) ? s : -s;
      return;
    }
    if (IsFull(b)) {
      scores[idx] = 0;
      return;
    }
  }
  scores[idx] = Evaluate(StonesOf(b, root_player),
                         StonesOf(b, 3 - root_player));
}

// Collapses one tree level: parent i = max or min of children 7i .. 7i + 6,
// skipping illegal (kInvalid) children.
__global__ void MinimaxReduceKernel(const int* children, int* parents,
                                    int num_parents, bool maximize) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_parents) return;
  int best = kInvalid;
  for (int c = 0; c < kWidth; ++c) {
    const int v = children[i * kWidth + c];
    if (v == kInvalid) continue;
    if (best == kInvalid || (maximize ? v > best : v < best)) best = v;
  }
  parents[i] = best;
}

int Blocks(int n) { return (n + kThreads - 1) / kThreads; }

// Picks the legal column with the highest score, preferring central columns.
void ChooseBest(MoveResult* result) {
  for (int col : kColumnOrder) {
    if (!result->legal[col]) continue;
    if (result->column < 0 || result->column_scores[col] > result->score) {
      result->column = col;
      result->score = result->column_scores[col];
    }
  }
}

}  // namespace

cudaError_t InitStrategies() {
  std::vector<uint64_t> windows;
  for (int col = 0; col < kWidth; ++col) {
    for (int row = 0; row < kHeight; ++row) {
      const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
      for (const auto& d : dirs) {
        const int end_col = col + 3 * d[0];
        const int end_row = row + 3 * d[1];
        if (end_col >= kWidth || end_row < 0 || end_row >= kHeight) continue;
        uint64_t w = 0;
        for (int i = 0; i < 4; ++i) {
          w |= CellMask(col + i * d[0], row + i * d[1]);
        }
        windows.push_back(w);
      }
    }
  }
  if (windows.size() != kNumWindows) return cudaErrorInvalidValue;
  return cudaMemcpyToSymbol(kWindows, windows.data(),
                            kNumWindows * sizeof(uint64_t));
}

MoveResult MonteCarloMove(const Board& board, const MonteCarloConfig& config) {
  MoveResult result;
  result.work_unit = "playouts";
  unsigned long long* d_totals = nullptr;
  CUDA_CHECK(cudaMalloc(&d_totals, kWidth * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_totals, 0, kWidth * sizeof(unsigned long long)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  const dim3 grid(config.blocks_per_move, kWidth);
  MonteCarloKernel<<<grid, kThreads>>>(board, config.playouts_per_thread,
                                       config.seed, d_totals);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&result.gpu_ms, start, stop));

  unsigned long long totals[kWidth];
  CUDA_CHECK(cudaMemcpy(totals, d_totals, sizeof(totals),
                        cudaMemcpyDeviceToHost));
  const int64_t per_column = static_cast<int64_t>(config.blocks_per_move) *
                             kThreads * config.playouts_per_thread;
  for (int col = 0; col < kWidth; ++col) {
    result.legal[col] = CanPlay(board, col);
    if (!result.legal[col]) continue;
    result.column_scores[col] = totals[col] / (2.0 * per_column);
    result.work += per_column;
  }
  ChooseBest(&result);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_totals);
  return result;
}

MoveResult MinimaxMove(const Board& board, int depth) {
  MoveResult result;
  result.work_unit = "leaves";
  depth = depth < 1 ? 1 : (depth > kMaxDepth ? kMaxDepth : depth);
  int num_leaves = 1;
  for (int i = 0; i < depth; ++i) num_leaves *= kWidth;

  int *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, num_leaves * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, num_leaves / kWidth * sizeof(int)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  MinimaxLeafKernel<<<Blocks(num_leaves), kThreads>>>(board, depth,
                                                      num_leaves, d_in);
  // Level d holds 7^d nodes. Reducing level d chooses the move at ply d - 1,
  // which the root player makes when d - 1 is even (maximize).
  int n = num_leaves;
  for (int d = depth; d > 1; --d) {
    const int parents = n / kWidth;
    MinimaxReduceKernel<<<Blocks(parents), kThreads>>>(d_in, d_out, parents,
                                                       (d - 1) % 2 == 0);
    std::swap(d_in, d_out);
    n = parents;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&result.gpu_ms, start, stop));

  int root_scores[kWidth];
  CUDA_CHECK(cudaMemcpy(root_scores, d_in, sizeof(root_scores),
                        cudaMemcpyDeviceToHost));
  for (int col = 0; col < kWidth; ++col) {
    result.legal[col] = root_scores[col] != kInvalid;
    result.column_scores[col] = result.legal[col] ? root_scores[col] : NAN;
  }
  result.work = num_leaves;
  ChooseBest(&result);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_out);
  return result;
}

}  // namespace c4
