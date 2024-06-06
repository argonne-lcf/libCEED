#include "shell_compiler.h"

#include <string>
#include <utility>
#include <vector>

#include "shell_command.h"

namespace prtc {
namespace {

std::string concatenateFlags(const std::vector<std::string>& flags) {
  std::string all_flags{};
  for (const auto& f : flags) {
    all_flags += f + " ";
  }
  // Remove last space
  if (!all_flags.empty()) all_flags.pop_back();
  return all_flags;
}

}  // namespace

ShellCompiler::ShellCompiler(const std::string& executable,
                             const std::string& output_flag,
                             const std::string& object_flag,
                             const std::string& pic_flag,
                             const std::string& dynamic_flag)
    : executable_{executable},
      output_flag_{output_flag},
      compile_flags_{object_flag + " " + pic_flag},
      link_flags_{dynamic_flag},
      compile_and_link_flags_{pic_flag + " " + dynamic_flag} {}

std::string ShellCompiler::executable() const { return executable_; }

std::string ShellCompiler::outputFlag() const { return output_flag_; }

std::string ShellCompiler::compileFlags() const { return compile_flags_; }

std::string ShellCompiler::linkFlags() const { return link_flags_; }

std::string ShellCompiler::compileAndLinkFlags() const {
  return compile_and_link_flags_;
}

std::pair<bool, std::string> ShellCompiler::compile(
    const std::string& source_path, const std::string& output_path,
    const std::vector<std::string>& options) const {
  std::string compile_command = executable_ + " " + compile_flags_ + " " +
                                concatenateFlags(options) + " " + source_path +
                                " " + output_flag_ + " " + output_path;
  ShellCommand sc(compile_command);
  return sc.result();
}

std::pair<bool, std::string> ShellCompiler::link(
    const std::string& source_path, const std::string& output_path,
    const std::vector<std::string>& options) const {
  std::string link_command = executable_ + " " + link_flags_ + " " +
                             concatenateFlags(options) + " " + source_path +
                             " " + output_flag_ + " " + output_path;
  ShellCommand sc(link_command);
  return sc.result();
}

std::pair<bool, std::string> ShellCompiler::compileAndLink(
    const std::string& source_path, const std::string& output_path,
    const std::vector<std::string>& options) const {
  std::string build_command = executable_ + " " + compile_and_link_flags_ +
                              " " + concatenateFlags(options) + " " +
                              source_path + " " + output_flag_ + " " +
                              output_path;
  ShellCommand sc(build_command);
  return sc.result();
}

}  // namespace prtc

// int main() {
//   using namespace prtc;
//   {
//     ShellCompiler compiler("icpx","-o","-c","-fPIC","-shared");
//     compiler.compile("hello.cpp","hello.o",{"-O2"});
//     compiler.link("hello.o","hello1.so",{"-O2"});
//     compiler.compileAndLink("hello.cpp","hello2.so",{"-O3"});
//   }
// }
