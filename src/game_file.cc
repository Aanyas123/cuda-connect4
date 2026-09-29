#include "game_file.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>

namespace c4 {
namespace {

std::string StatePath(const std::string& dir) { return dir + "/state.txt"; }
std::string MovesPath(const std::string& dir) { return dir + "/moves.csv"; }

}  // namespace

FileLock::FileLock(const std::string& path)
    : fd_(open(path.c_str(), O_CREAT | O_RDWR, 0644)) {
  if (fd_ >= 0 && flock(fd_, LOCK_EX) != 0) {
    close(fd_);
    fd_ = -1;
  }
}

FileLock::~FileLock() {
  if (fd_ >= 0) {
    flock(fd_, LOCK_UN);
    close(fd_);
  }
}

std::string LockPath(const std::string& dir) { return dir + "/lock"; }

bool InitGame(const std::string& dir) {
  if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) return false;
  FileLock lock(LockPath(dir));
  if (!lock.ok() || !WriteState(dir, GameState())) return false;
  std::ofstream log(MovesPath(dir), std::ios::trunc);
  log << "move,player,strategy,column,score,gpu_ms,work\n";
  return static_cast<bool>(log);
}

bool ReadState(const std::string& dir, GameState* state) {
  std::ifstream in(StatePath(dir));
  if (!in) return false;
  GameState s;
  bool has_turn = false;
  std::string line;
  while (std::getline(in, line)) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = line.substr(0, eq);
    const std::string value = line.substr(eq + 1);
    if (key == "turn") {
      s.turn = std::atoi(value.c_str());
      has_turn = true;
    } else if (key == "status") {
      s.status = value;
    } else if (key == "moves") {
      s.moves = value;
    } else if (key == "p1") {
      s.p1_name = value;
    } else if (key == "p2") {
      s.p2_name = value;
    }
  }
  if (!has_turn) return false;
  *state = s;
  return true;
}

bool WriteState(const std::string& dir, const GameState& state) {
  const std::string path = StatePath(dir);
  const std::string tmp = path + ".tmp";
  std::ofstream out(tmp, std::ios::trunc);
  out << "turn=" << state.turn << "\n"
      << "status=" << state.status << "\n"
      << "moves=" << state.moves << "\n"
      << "p1=" << state.p1_name << "\n"
      << "p2=" << state.p2_name << "\n";
  out.close();
  if (!out) return false;
  return std::rename(tmp.c_str(), path.c_str()) == 0;
}

bool AppendMoveLog(const std::string& dir, const std::string& line) {
  std::ofstream log(MovesPath(dir), std::ios::app);
  log << line << "\n";
  return static_cast<bool>(log);
}

}  // namespace c4
