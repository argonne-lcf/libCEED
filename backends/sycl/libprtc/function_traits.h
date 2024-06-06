#pragma once

#include <functional>

namespace prtc {

template <class F>
class function_traits;

template <class R, class... Args>
class function_traits<std::function<R(Args...)>> {
 public:
  static std::function<R(Args...)> cast(void* p) {
    return reinterpret_cast<R (*)(Args...)>(p);
  }
};

template <class F>
F function_cast(void* p) {
  return function_traits<F>::cast(p);
}

}  // namespace prtc
