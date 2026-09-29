// Connect Four between two GPU competitors.
//
// Each competitor is a separate process (optionally on its own GPU) that
// waits for its turn in a shared game directory, decides a move with a CUDA
// strategy, and writes the move back under a file lock. A third "watch"
// process draws the board after every move.
//
//   connect4 --init  --game_dir g
//   connect4 --play 1 --strategy mc      --game_dir g &
//   connect4 --play 2 --strategy minimax --game_dir g &
//   connect4 --watch --game_dir g

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

#include "board.h"
#include "board_util.h"
#include "cuda_check.cuh"
#include "game_file.h"
#include "strategies.cuh"

namespace {

using Clock = std::chrono::steady_clock;

enum class Mode { kNone, kInit, kPlay, kWatch };

struct Options {
  Mode mode = Mode::kNone;
  std::string game_dir;
  std::string strategy;
  std::string log_path;
  int player = 0;
  int depth = 7;
  int blocks = 64;
  int playouts = 4;
  uint64_t seed = 1;
  int device = 0;
  int move_delay_ms = 0;
  int timeout_s = 120;
  bool help = false;
};

void PrintUsage(const char* prog) {
  std::printf(
      "Usage:\n"
      "  %s --init  --game_dir DIR\n"
      "  %s --play N --strategy mc|minimax --game_dir DIR [options]\n"
      "  %s --watch --game_dir DIR [--log FILE]\n"
      "Options:\n"
      "  --depth D          minimax search depth, 1-8 (default 7)\n"
      "  --blocks N         mc blocks of 256 threads per column (default 64)\n"
      "  --playouts N       mc playouts per thread (default 4)\n"
      "  --seed S           mc random seed (default 1)\n"
      "  --device N         CUDA device for this player (default 0)\n"
      "  --move_delay_ms N  pause after each move so viewers can follow\n"
      "  --timeout_s N      give up if the game stalls this long (default 120)\n"
      "  --log FILE         (watch) also write plain-text boards to FILE\n",
      prog, prog, prog);
}

bool ParseArgs(int argc, char** argv, Options* opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    auto int_value = [&]() { return std::atoi(argv[++i]); };
    if (arg == "--help" || arg == "-h") {
      opts->help = true;
      return true;
    } else if (arg == "--init") {
      opts->mode = Mode::kInit;
    } else if (arg == "--watch") {
      opts->mode = Mode::kWatch;
    } else if (arg == "--play" && has_value) {
      opts->mode = Mode::kPlay;
      opts->player = int_value();
    } else if (arg == "--game_dir" && has_value) {
      opts->game_dir = argv[++i];
    } else if (arg == "--strategy" && has_value) {
      opts->strategy = argv[++i];
    } else if (arg == "--log" && has_value) {
      opts->log_path = argv[++i];
    } else if (arg == "--depth" && has_value) {
      opts->depth = int_value();
    } else if (arg == "--blocks" && has_value) {
      opts->blocks = int_value();
    } else if (arg == "--playouts" && has_value) {
      opts->playouts = int_value();
    } else if (arg == "--seed" && has_value) {
      opts->seed = std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "--device" && has_value) {
      opts->device = int_value();
    } else if (arg == "--move_delay_ms" && has_value) {
      opts->move_delay_ms = int_value();
    } else if (arg == "--timeout_s" && has_value) {
      opts->timeout_s = int_value();
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n",
                   arg.c_str());
      return false;
    }
  }
  if (opts->mode == Mode::kNone || opts->game_dir.empty()) return false;
  if (opts->mode == Mode::kPlay) {
    return (opts->player == 1 || opts->player == 2) &&
           (opts->strategy == "mc" || opts->strategy == "minimax") &&
           opts->depth >= 1 && opts->depth <= c4::kMaxDepth &&
           opts->blocks >= 1 && opts->playouts >= 1 && opts->device >= 0;
  }
  return true;
}

void SleepMs(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

double SecondsSince(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

std::string FormatScores(const c4::MoveResult& r) {
  std::string out = "[";
  char buf[32];
  for (int col = 0; col < c4::kWidth; ++col) {
    if (!r.legal[col]) {
      std::snprintf(buf, sizeof(buf), "%s-", col ? " " : "");
    } else {
      std::snprintf(buf, sizeof(buf), "%s%.3g", col ? " " : "",
                    r.column_scores[col]);
    }
    out += buf;
  }
  return out + "]";
}

std::string ResultText(const c4::GameState& s) {
  const int total = static_cast<int>(s.moves.size());
  if (s.status == "p1_wins") {
    return "Player 1 (X, " + s.p1_name + ") wins after " +
           std::to_string(total) + " moves";
  }
  if (s.status == "p2_wins") {
    return "Player 2 (O, " + s.p2_name + ") wins after " +
           std::to_string(total) + " moves";
  }
  return "Draw: the board is full";
}

// ---------------------------------------------------------------------------
// Player process.
// ---------------------------------------------------------------------------

int RunPlayer(const Options& opts) {
  CUDA_CHECK(cudaSetDevice(opts.device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));
  CUDA_CHECK(c4::InitStrategies());

  char tag[64];
  std::snprintf(tag, sizeof(tag), "[P%d %s]", opts.player,
                opts.strategy.c_str());
  std::printf("%s GPU %d: %s (compute %d.%d, %d SMs)\n", tag, opts.device,
              prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  const std::string lock_path = c4::LockPath(opts.game_dir);

  // Join the game by recording this player's strategy name.
  {
    c4::FileLock lock(lock_path);
    c4::GameState state;
    if (!lock.ok() || !c4::ReadState(opts.game_dir, &state)) {
      std::fprintf(stderr, "%s cannot read game in %s (run --init first)\n",
                   tag, opts.game_dir.c_str());
      return 1;
    }
    (opts.player == 1 ? state.p1_name : state.p2_name) = opts.strategy;
    if (!c4::WriteState(opts.game_dir, state)) return 1;
  }

  size_t last_seen = 0;
  Clock::time_point last_progress = Clock::now();
  while (true) {
    c4::GameState state;
    {
      c4::FileLock lock(lock_path);
      if (!lock.ok() || !c4::ReadState(opts.game_dir, &state)) {
        std::fprintf(stderr, "%s lost the game state\n", tag);
        return 1;
      }
    }
    if (state.status != "playing") {
      std::printf("%s game over: %s\n", tag, ResultText(state).c_str());
      return 0;
    }
    if (state.moves.size() != last_seen) {
      last_seen = state.moves.size();
      last_progress = Clock::now();
    }
    if (state.turn != opts.player) {
      if (SecondsSince(last_progress) > opts.timeout_s) {
        std::fprintf(stderr, "%s opponent did not move in %d s\n", tag,
                     opts.timeout_s);
        return 1;
      }
      SleepMs(5);
      continue;
    }

    // Our turn: rebuild the position and let the GPU decide.
    c4::Board board;
    if (!c4::BuildBoard(state.moves, &board)) {
      std::fprintf(stderr, "%s invalid move list '%s'\n", tag,
                   state.moves.c_str());
      return 1;
    }
    c4::MoveResult move;
    if (opts.strategy == "mc") {
      c4::MonteCarloConfig config;
      config.blocks_per_move = opts.blocks;
      config.playouts_per_thread = opts.playouts;
      config.seed = opts.seed * 1000003ULL + board.moves;
      move = c4::MonteCarloMove(board, config);
    } else {
      move = c4::MinimaxMove(board, opts.depth);
    }
    if (move.column < 0) {
      std::fprintf(stderr, "%s found no legal move\n", tag);
      return 1;
    }

    c4::Play(&board, move.column);
    std::string status = "playing";
    if (c4::HasFour(c4::LastMoverStones(board))) {
      status = opts.player == 1 ? "p1_wins" : "p2_wins";
    } else if (c4::IsFull(board)) {
      status = "draw";
    }

    // Commit the move under the lock, checking nobody else moved meanwhile.
    {
      c4::FileLock lock(lock_path);
      c4::GameState now;
      if (!lock.ok() || !c4::ReadState(opts.game_dir, &now) ||
          now.moves != state.moves || now.turn != opts.player) {
        std::fprintf(stderr, "%s game state changed during my turn\n", tag);
        return 1;
      }
      now.moves += static_cast<char>('0' + move.column);
      now.turn = 3 - opts.player;
      now.status = status;
      char line[160];
      std::snprintf(line, sizeof(line), "%d,%d,%s,%d,%.4f,%.3f,%lld",
                    board.moves, opts.player, opts.strategy.c_str(),
                    move.column, move.score, move.gpu_ms,
                    static_cast<long long>(move.work));
      if (!c4::WriteState(opts.game_dir, now) ||
          !c4::AppendMoveLog(opts.game_dir, line)) {
        std::fprintf(stderr, "%s cannot write game state\n", tag);
        return 1;
      }
    }
    std::printf("%s move %2d: column %d  score %-8.4g %lld %s in %.2f ms  "
                "column scores %s\n",
                tag, board.moves, move.column, move.score,
                static_cast<long long>(move.work), move.work_unit,
                move.gpu_ms, FormatScores(move).c_str());
    std::fflush(stdout);
    if (opts.move_delay_ms > 0) SleepMs(opts.move_delay_ms);
  }
}

// ---------------------------------------------------------------------------
// Watcher process.
// ---------------------------------------------------------------------------

int RunWatcher(const Options& opts) {
  const bool color = isatty(STDOUT_FILENO);
  std::ofstream log;
  if (!opts.log_path.empty()) log.open(opts.log_path);
  auto emit = [&](const std::string& title, const c4::Board& board,
                  int last_col) {
    std::printf("%s\n%s\n", title.c_str(),
                c4::RenderBoard(board, color, last_col).c_str());
    std::fflush(stdout);
    if (log.is_open()) {
      log << title << "\n" << c4::RenderBoard(board, false, last_col) << "\n";
    }
  };

  emit("Connect Four: Player 1 = X (red), Player 2 = O (yellow)", c4::Board(),
       -1);
  size_t shown = 0;
  Clock::time_point last_progress = Clock::now();
  while (true) {
    c4::GameState state;
    bool ok;
    {
      c4::FileLock lock(c4::LockPath(opts.game_dir));
      ok = lock.ok() && c4::ReadState(opts.game_dir, &state);
    }
    if (ok) {
      while (shown < state.moves.size()) {
        ++shown;
        c4::Board board;
        c4::BuildBoard(state.moves.substr(0, shown), &board);
        const int player = (shown % 2 == 1) ? 1 : 2;
        const std::string& name = player == 1 ? state.p1_name : state.p2_name;
        const int col = state.moves[shown - 1] - '0';
        emit("Move " + std::to_string(shown) + ": Player " +
                 std::to_string(player) + (player == 1 ? " (X, " : " (O, ") +
                 name + ") plays column " + std::to_string(col),
             board, col);
        last_progress = Clock::now();
      }
      if (state.status != "playing") {
        const std::string result = "RESULT: " + ResultText(state);
        std::printf("%s\n", result.c_str());
        if (log.is_open()) log << result << "\n";
        return 0;
      }
    }
    if (SecondsSince(last_progress) > opts.timeout_s) {
      std::fprintf(stderr, "Watcher: no move for %d s, giving up\n",
                   opts.timeout_s);
      return 1;
    }
    SleepMs(10);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  const bool args_ok = ParseArgs(argc, argv, &opts);
  if (opts.help) {
    PrintUsage(argv[0]);
    return 0;
  }
  if (!args_ok) {
    PrintUsage(argv[0]);
    return 1;
  }
  switch (opts.mode) {
    case Mode::kInit:
      if (!c4::InitGame(opts.game_dir)) {
        std::fprintf(stderr, "Cannot create game in %s\n",
                     opts.game_dir.c_str());
        return 1;
      }
      std::printf("New game in %s\n", opts.game_dir.c_str());
      return 0;
    case Mode::kPlay:
      return RunPlayer(opts);
    case Mode::kWatch:
      return RunWatcher(opts);
    default:
      return 1;
  }
}
