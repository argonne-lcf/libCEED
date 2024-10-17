#pragma once

#include <memory>
#include <optional>
#include <string>
#include <functional>

#include "function_traits.h"
#include <functional>

namespace prtc {

class DynamicLibrary : public std::enable_shared_from_this<DynamicLibrary> {
 public:
  static std::shared_ptr<DynamicLibrary> open(const std::string& path);

  explicit DynamicLibrary(const std::string& path);

  ~DynamicLibrary();

  std::shared_ptr<DynamicLibrary> share();

  std::string path() const;

  template <class F>
  F getFunction(const std::string& name) {
    // return function_cast<F>(getSymbol(name));
    return reinterpret_cast<F>(getSymbol(name));
  }

  void * getFunction2(const std::string& name) {
    return getSymbol(name);
  }

 private:
  void* getSymbol(const std::string& name) const;

  std::string path_;
  void* handle_;
};

}  // namespace prtc
