#include "dynamic_library.h"

#include <dlfcn.h>

#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace prtc {

std::shared_ptr<DynamicLibrary> DynamicLibrary::open(const std::string& path) {
  std::cout<<"\n Creating Module from path "<<path<<std::endl;
  std::shared_ptr<DynamicLibrary> dyn_lib = std::make_shared<DynamicLibrary>(path);
  std::cout<<"\n Module created from path "<<dyn_lib->path()<<std::endl;
  return dyn_lib;
}

DynamicLibrary::DynamicLibrary(const std::string& path)
    : path_{path}, handle_{dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)} {
  if (!handle_) {
    char* dlerror_message = dlerror();
    std::string error_message =
        "DynamicLibrary: failed to load " + path + "\n" + dlerror_message;
    throw std::runtime_error(error_message);
  }
  std::cout<<"\n Module created from path "<<path<<std::endl;
}

DynamicLibrary::~DynamicLibrary() { dlclose(handle_); }

std::shared_ptr<prtc::DynamicLibrary> DynamicLibrary::share() {
  return shared_from_this();
}

std::string DynamicLibrary::path() const { return path_; }

void* DynamicLibrary::getSymbol(const std::string& name) const {
  std::cout<<"\n Entered GetSymbol\n";
  std::cout<<"Looking for symbol " << name << " in path: " << path_ <<"\n";
  void* symbol = dlsym(handle_, name.c_str());
  std::cout<<"\n Received Symbol\n";
  if (!symbol) {
    char* dlerror_message = dlerror();
    std::cout<<"Looking in path: " << path_ <<"\n";
    std::string error_message = "DynamicLibrary: failed to find symbol " +
                                name + " in " + path_ + "\n" + dlerror_message;
    throw std::runtime_error(error_message);
  }
  return symbol;
}

}  // namespace prtc

// int main() {
//   // auto library =
//   prtc::DynamicLibrary::open("./libtest-dynamic-library.so");
//   prtc::DynamicLibrary lib("./libtest-dynamic-library.so");

//   std::shared_ptr<prtc::DynamicLibrary> library = lib.share();
//   // auto pl = library->share();

//   // std::cout << "pl.use_count()" << pl.use_count() << "\n";

//   auto axpy = library->getFunction<std::function<int(int, int, int)>
//   >("axpy"); try {
//     auto gemm = library->getFunction<std::function<int(int)> >("gemm");
//   } catch (const std::exception& e) {
//     std::cout << e.what() << std::endl;
//   }

//   const int alpha = 2;
//   const int x = 1;
//   const int y = 2;
//   const int expected = alpha * x + y;
//   const int actual = axpy(alpha, x, y);

//   std::cout << "expected: " << expected << "\n";
//   std::cout << "actual: " << actual << "\n";

//   return 0;
// }
