#pragma once

#include <string>
#include <utility>
#include <vector>

namespace prtc {

std::string concatenateFlags(const std::vector<std::string>& flags);

class ShellCompiler {
 public:
  ShellCompiler(const std::string& executable, const std::string& output_flag,
                const std::string& object_flag, const std::string& pic_flag,
                const std::string& dynamic_flag);

  std::string executable() const;
  std::string outputFlag() const;
  std::string compileFlags() const;
  std::string linkFlags() const;
  std::string compileAndLinkFlags() const;

  std::pair<bool, std::string> compile(
      const std::string& source_path, const std::string& output_path,
      const std::vector<std::string>& options = {}) const;

  std::pair<bool, std::string> link(
      const std::string& source_path, const std::string& output_path,
      const std::vector<std::string>& options = {}) const;

  std::pair<bool, std::string> compileAndLink(
      const std::string& source_path, const std::string& output_path,
      const std::vector<std::string>& options = {}) const;

 private:
  std::string executable_;
  std::string output_flag_;
  std::string compile_flags_;
  std::string link_flags_;
  std::string compile_and_link_flags_;
};

}  // namespace prtc
