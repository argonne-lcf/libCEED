// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-compile.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <level_zero/ze_api.h>

#include <cstdlib>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <sycl/sycl.hpp>

#include "ceed-sycl-common.hpp"
#include "libprtc/prtc.h"

//------------------------------------------------------------------------------
// Add defined constants at the beginning of kernel source
//------------------------------------------------------------------------------
static int CeedJitAddDefinitions_Sycl(Ceed ceed, const std::string &kernel_source, std::string &jit_source,
                                      const std::map<std::string, CeedInt> &constants = {}) {
  std::ostringstream oss;

  const char *jit_defs_path, *jit_defs_source;
  const char *sycl_jith_path = "ceed/jit-source/sycl/sycl-jit.h";

  oss << "#include<sycl/sycl.hpp>\n\n";

  // Prepend defined constants
  for (const auto &[name, value] : constants) {
    oss << "#define " << name << " " << value << "\n";
  }

  // libCeed definitions for Sycl Backends
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, sycl_jith_path, &jit_defs_path));
  {
    char *source;

    CeedCallBackend(CeedLoadSourceToBuffer(ceed, jit_defs_path, &source));
    jit_defs_source = source;
  }

  oss << jit_defs_source << "\n";

  CeedCallBackend(CeedFree(&jit_defs_path));
  CeedCallBackend(CeedFree(&jit_defs_source));

  // Append kernel_source
  oss << "\n" << kernel_source;

  jit_source = oss.str();
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// TODO: Add architecture flags, optimization flags
//------------------------------------------------------------------------------
static inline int CeedJitGetFlags_Sycl(std::vector<std::string> &flags) {

  // flags = {std::string("-cl-std=CL3.0"), std::string("-Dint32_t=int")};
  flags = {std::string("-fsycl"), std::string("-fno-sycl-id-queries-fit-in-int")};
  // TODO : Add AOT flags and other optimization flags
  // flags.push_back(std::string("-O3"));
  // flags.push_back(std::string("-fsycl-targets=spir64_gen -Xsycl-target-backend \"-device pvc\" "))
  
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compile sycl source to a shared library
// TODO: Check if source, module, etc. already exists
//------------------------------------------------------------------------------
static inline int CeedJitCompileSource_Sycl(Ceed ceed, const sycl::device &sycl_device, const std::string &kernel_source, std::string& output_path,
                                            const std::vector<std::string> flags = {}) {

  // Get cache path from env variable
  std::string cache_root;
  // TODO: Add default directory to current working directory
  if(std::getenv("CEED_CACHE_DIR")) {
    cache_root = std::string(std::getenv("CEED_CACHE_DIR")) + "/.ceed/cache";
  } else {
    cache_root = std::string(std::getenv("PWD")) + "/.ceed/cache";
  }

  // Generate kernel hash
  // E.g., see https://intel.github.io/llvm-docs/design/KernelProgramCache.html
  // An example of directory structure can be found here:
  // https://intel.github.io/llvm-docs/design/KernelProgramCache.html#persistent-cache-storage-structure

  // UU: Plan to use cache storage structure as : <cache_root>/<compiler_hash>/<build_options>/<kernel_name>/<kernel_source>

  // Hash kernel name and source
  std::hash<std::string> string_hash;
  // size_t kernel_name_hash   = string_hash(get_kernel_name) ! UU : TODO LATER
  size_t kernel_source_hash = string_hash(kernel_source);

  // Hash compilation flags
  std::vector<std::string> copy_flags = flags;
  std::sort(copy_flags.begin(), copy_flags.end());
  std::string all_flags = prtc::concatenateFlags(copy_flags);
  size_t build_options_hash = string_hash(all_flags);

  // Hash compiler version
  prtc::ShellCommand command("icpx --version");
  const auto [success, compiler_version] = command.result();
  if (!success) return CeedError((ceed), CEED_ERROR_BACKEND, compiler_version.c_str());
  size_t compiler_hash = string_hash(compiler_version);

  // Determine file paths for source and binaries based on hashes
  std::string cache_path = cache_root + "/" + std::to_string(compiler_hash) + "/" + std::to_string(build_options_hash) + "/" + std::to_string(kernel_source_hash) + "/";
  std::string source_file_path = cache_path + "source.cpp";
  std::string object_file_path = cache_path + "binary.so";
  std::string mkdir_command = std::string("mkdir -p ") + cache_path;
  prtc::ShellCommand make_dir(mkdir_command);
  auto [mkdir_success, mkdir_message] = make_dir.result();
  if(!mkdir_success) return CeedError((ceed), CEED_ERROR_BACKEND, mkdir_message.c_str());

  // Write source string to file
  std::ofstream source_file;
  source_file.open(source_file_path);
  source_file << kernel_source;
  source_file.close();

  // TODO: Get compiler-path and flags from env or some other means
  prtc::ShellCompiler compiler("icpx","-o","-c","-fPIC","-shared");
  const auto [build_success, build_message] = compiler.compileAndLink(source_file_path,object_file_path,flags);
  // Q: Should we always output the compiler output in verbose/debug mode?
  if (!build_success) return CeedError((ceed), CEED_ERROR_BACKEND, build_message.c_str());
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Load (compile) SPIR-V source and wrap in sycl kernel_bundle
// ------------------------------------------------------------------------------
static int CeedLoadModule_Sycl(Ceed ceed, const sycl::context &sycl_context, const sycl::device &sycl_device, const std::string& path,
                               SyclModule_t* sycl_module) {
  try {
    *sycl_module =  prtc::DynamicLibrary::open(path);
    std::string check_path = (*sycl_module)->path();
    std::cout<<"\n Module created from path "<<check_path<<std::endl;
  } catch (const std::exception& e) {
    return CeedError((ceed), CEED_ERROR_BACKEND, e.what());
  }
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Compile kernel source to a shared library
// ------------------------------------------------------------------------------
int CeedBuildModule_Sycl(Ceed ceed, const std::string &kernel_source, SyclModule_t* sycl_module, const std::map<std::string, CeedInt> &constants) {
  Ceed_Sycl               *data;
  std::string              jit_source;
  std::string              module_path;
  std::vector<std::string> flags;

  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedJitAddDefinitions_Sycl(ceed, kernel_source, jit_source, constants));
  CeedCallBackend(CeedJitGetFlags_Sycl(flags));

  CeedCallBackend(CeedJitCompileSource_Sycl(ceed, data->sycl_device, jit_source, module_path, flags));
  
  CeedCallBackend(CeedLoadModule_Sycl(ceed, data->sycl_context, data->sycl_device, module_path, sycl_module));
  
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Get a sycl kernel from an existing module
// ------------------------------------------------------------------------------
// template <class SyclKernel_t>
// int CeedGetKernel_Sycl(Ceed ceed, const SyclModule_t sycl_module, const std::string &kernel_name, SyclKernel_t *sycl_kernel) {
//   try {
//     *sycl_kernel = sycl_module->getFunction<SyclKernel_t>(kernel_name);
//   } catch (const std::exception& e) {
//     return CeedError((ceed), CEED_ERROR_BACKEND, e.what());
//   }
//   return CEED_ERROR_SUCCESS;
// }

//------------------------------------------------------------------------------
// Run SYCL kernel for spatial dimension with shared memory
//------------------------------------------------------------------------------
template <class SyclKernel_t>
int CeedRunKernelDimSharedSycl(Ceed ceed, SyclKernel_t *kernel, const int grid_size, const int block_size_x, const int block_size_y,
                               const int block_size_z, const int shared_mem_size, void **kernel_args) {
  sycl::range<3>    local_range(block_size_z, block_size_y, block_size_x);
  sycl::range<3>    global_range(grid_size * block_size_z, block_size_y, block_size_x);
  sycl::nd_range<3> kernel_range(global_range, local_range);

  //-----------
  // Order queue
  // Ceed_Sycl *ceed_Sycl;

  // CeedCallBackend(CeedGetData(ceed, &ceed_Sycl));
  // sycl::event e = ceed_Sycl->sycl_queue.ext_oneapi_submit_barrier();

  // ceed_Sycl->sycl_queue.submit([&](sycl::handler &cgh) {
  //   cgh.depends_on(e);
  //   cgh.set_args(*kernel_args);
  //   cgh.parallel_for(kernel_range, *kernel);
  // });
  return CEED_ERROR_SUCCESS;
}
