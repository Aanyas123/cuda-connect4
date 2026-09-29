// Shared game directory used by the two player processes to take turns.
//
// <dir>/state.txt  key=value lines: turn, status, moves, p1, p2
// <dir>/lock       flock() target; hold it while reading or writing state
// <dir>/moves.csv  one line per move, appended by the player who moved
#ifndef CONNECT4_GAME_FILE_H_
#define CONNECT4_GAME_FILE_H_

#include <string>

namespace c4 {

struct GameState {
  int turn = 1;                    // Player (1 or 2) who must move next.
  std::string status = "playing";  // playing | p1_wins | p2_wins | draw
  std::string moves;               // One digit '0'-'6' per move played.
  std::string p1_name;             // Strategy names, set when players join.
  std::string p2_name;
};

// Exclusive advisory lock on a file, released when the object is destroyed.
class FileLock {
 public:
  explicit FileLock(const std::string& path);
  ~FileLock();
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;

  bool ok() const { return fd_ >= 0; }

 private:
  int fd_;
};

std::string LockPath(const std::string& dir);

// Creates |dir| (one level) with a fresh game state and move log.
bool InitGame(const std::string& dir);

bool ReadState(const std::string& dir, GameState* state);

// Writes to a temporary file and renames it over state.txt, so readers never
// see a half-written file.
bool WriteState(const std::string& dir, const GameState& state);

bool AppendMoveLog(const std::string& dir, const std::string& line);

}  // namespace c4

#endif  // CONNECT4_GAME_FILE_H_
