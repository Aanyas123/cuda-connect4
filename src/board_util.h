// Host-side helpers: rebuilding a board from a move list and drawing it.
#ifndef CONNECT4_BOARD_UTIL_H_
#define CONNECT4_BOARD_UTIL_H_

#include <string>

#include "board.h"

namespace c4 {

// Replays |moves| (one digit '0'-'6' per move) from the empty board.
// Returns false if a move is illegal or is played after the game ended.
bool BuildBoard(const std::string& moves, Board* board);

// Draws the board as text. Player 1 is X, player 2 is O. |last_col| (or -1)
// is marked with a 'v' above the column. |color| adds ANSI colors.
std::string RenderBoard(const Board& board, bool color, int last_col);

}  // namespace c4

#endif  // CONNECT4_BOARD_UTIL_H_
