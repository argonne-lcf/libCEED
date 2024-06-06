#include "shell_command.h"

#include <stdio.h>

#include <iostream>
#include <string>
#include <system_error>
#include <utility>

namespace {

std::string get_system_error_message() {
  return std::make_error_condition(std::errc(errno)).message();
}

std::string read_stream(std::FILE* stream) {
  std::string output;
  for (char buffer[16]; std::fgets(buffer, sizeof(buffer), stream);) {
    output += buffer;
  }
  return output;
}

std::pair<bool, std::string> pipe_open(const std::string& command) {
  std::FILE* pipe_stream = popen(command.c_str(), "r");
  if (!pipe_stream) {
    return std::pair<bool, std::string>(false, get_system_error_message());
  } else {
    std::string output = read_stream(pipe_stream);
    const int status = pclose(pipe_stream);
    const bool success = !status;
    const std::string message =
        (0 > status) ? get_system_error_message() : output;
    return std::pair<bool, std::string>(success, message);
  }
}

}  // namespace

namespace prtc {

ShellCommand::ShellCommand(std::string command)
    : command_{command}, result_{pipe_open(command)} {}

std::string ShellCommand::command() const { return command_; }

std::pair<bool, std::string> ShellCommand::result() const { return result_; }

}  // namespace prtc

// int main() {
//   using namespace prtc;
//   {
//     ShellCommand command("echo \"Hello World!\"");
//     const auto [success, message] = command.result();
//     std::cout << (success ? "Success: " : "Error: ") << message << std::endl;
//   }

//   {
//     ShellCommand command("ls -lh");
//     const auto [success, message] = command.result();
//     std::cout << (success ? "Success: " : "Error: ") << message << std::endl;
//   }
//   return 0;
// }
