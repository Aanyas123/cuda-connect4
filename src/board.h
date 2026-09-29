// Connect Four bitboard shared by host and device code.
//
// Each column uses kHeight + 1 = 7 bits (6 cells plus one spare bit), so
// cell (col, row) is bit row + col * 7, with row 0 at the bottom. A position
// is stored as two 64-bit masks: |current| holds the stones of the player to
// move and |mask| holds every stone on the board.
#ifndef CONNECT4_BOARD_H_
#define CONNECT4_BOARD_H_

#include <cstdint>

#ifdef __CUDACC__
#define C4_HD __host__ __device__
#else
#define C4_HD
#endif

namespace c4 {

constexpr int kWidth = 7;
constexpr int kHeight = 6;
constexpr int kCells = kWidth * kHeight;

struct Board {
  uint64_t current = 0;  // Stones of the player to move.
  uint64_t mask = 0;     // All stones.
  int moves = 0;         // Stones played so far.
};

C4_HD inline uint64_t BottomMask(int col) {
  return 1ULL << (col * (kHeight + 1));
}

C4_HD inline uint64_t TopMask(int col) {
  return 1ULL << (kHeight - 1 + col * (kHeight + 1));
}

C4_HD inline uint64_t CellMask(int col, int row) {
  return 1ULL << (row + col * (kHeight + 1));
}

C4_HD inline bool CanPlay(const Board& b, int col) {
  return col >= 0 && col < kWidth && (b.mask & TopMask(col)) == 0;
}

// Drops a stone for the player to move. The caller must check CanPlay().
C4_HD inline void Play(Board* b, int col) {
  b->current ^= b->mask;  // Now holds the opponent's stones.
  b->mask |= b->mask + BottomMask(col);
  ++b->moves;
}

// True if |stones| contain four in a row in any direction.
C4_HD inline bool HasFour(uint64_t stones) {
  uint64_t m = stones & (stones >> (kHeight + 1));  // Horizontal.
  if (m & (m >> (2 * (kHeight + 1)))) return true;
  m = stones & (stones >> kHeight);  // Diagonal "\".
  if (m & (m >> (2 * kHeight))) return true;
  m = stones & (stones >> (kHeight + 2));  // Diagonal "/".
  if (m & (m >> (2 * (kHeight + 2)))) return true;
  m = stones & (stones >> 1);  // Vertical.
  return (m & (m >> 2)) != 0;
}

// Stones of the player who made the most recent move.
C4_HD inline uint64_t LastMoverStones(const Board& b) {
  return b.current ^ b.mask;
}

C4_HD inline bool IsFull(const Board& b) { return b.moves == kCells; }

// Player (1 or 2) whose turn it is. Player 1 always moves first.
C4_HD inline int PlayerToMove(const Board& b) {
  return (b.moves % 2 == 0) ? 1 : 2;
}

C4_HD inline uint64_t StonesOf(const Board& b, int player) {
  return PlayerToMove(b) == player ? b.current : b.current ^ b.mask;
}

C4_HD inline bool IsWinningMove(const Board& b, int col) {
  Board next = b;
  Play(&next, col);
  return HasFour(LastMoverStones(next));
}

}  // namespace c4

#endif  // CONNECT4_BOARD_H_
