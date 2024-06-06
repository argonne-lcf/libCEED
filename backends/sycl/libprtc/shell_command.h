#pragma once

#include <string>
#include <utility>

namespace prtc {

class ShellCommand {
 public:
  ShellCommand() = delete;
  explicit ShellCommand(std::string command);

  std::string command() const;
  std::pair<bool, std::string> result() const;

 private:
  std::string command_;
  std::pair<bool, std::string> result_;
};

}  // namespace prtc
