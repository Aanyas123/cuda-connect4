#include "board_util.h"

namespace c4 {

bool BuildBoard(const std::string& moves, Board* board) {
  Board b;
  for (char ch : moves) {
    const int col = ch - '0';
    if (HasFour(LastMoverStones(b)) || !CanPlay(b, col)) return false;
    Play(&b, col);
  }
  *board = b;
  return true;
}

std::string RenderBoard(const Board& board, bool color, int last_col) {
  const uint64_t p1 = StonesOf(board, 1);
  std::string out;
  if (last_col >= 0 && last_col < kWidth) {
    out += std::string(1 + 2 * last_col, ' ') + "v\n";
  }
  out += " 0 1 2 3 4 5 6\n";
  for (int row = kHeight - 1; row >= 0; --row) {
    out += "|";
    for (int col = 0; col < kWidth; ++col) {
      const uint64_t cell = CellMask(col, row);
      if ((board.mask & cell) == 0) {
        out += ".";
      } else if (p1 & cell) {
        out += color ? "\033[1;31mX\033[0m" : "X";
      } else {
        out += color ? "\033[1;33mO\033[0m" : "O";
      }
      out += (col + 1 < kWidth) ? " " : "|\n";
    }
  }
  out += "+-------------+\n";
  return out;
}

}  // namespace c4
