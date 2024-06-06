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
#include <sstream>
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

  flags = {std::string("-cl-std=CL3.0"), std::string("-Dint32_t=int")};
  
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compile sycl source to a shared library
// TODO: Check if source, module, etc. already exists
//------------------------------------------------------------------------------
static inline int CeedJitCompileSource_Sycl(Ceed ceed, const sycl::device &sycl_device, const std::string &kernel_source, std::string& output_path,
                                            const std::vector<std::string> &flags = {}) {

  // Get cache path from env variable
  const char* cache_path = std::getenv("CEED_CACHE_DIR");

  // Generate kernel hash
  // E.g., see https://intel.github.io/llvm-docs/design/KernelProgramCache.html
  // An example of directory structure can be found here:
  // https://intel.github.io/llvm-docs/design/KernelProgramCache.html#persistent-cache-storage-structure

  // Write source string to file
  std::string source_path;

  // TODO: Get compiler-path and flags from env or some other means
  prtc::ShellCompiler compiler("icpx","-o","-c","-fPIC","-shared");
  const auto [success, message] = compiler.compileAndLink(source_path,output_path,flags);
  // Q: Should we always output the compiler output in verbose/debug mode?
  if (!success) return CeedError((ceed), CEED_ERROR_BACKEND, message);
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Load (compile) SPIR-V source and wrap in sycl kernel_bundle
// ------------------------------------------------------------------------------
static int CeedLoadModule_Sycl(Ceed ceed, const sycl::context &sycl_context, const sycl::device &sycl_device, const std::string& path,
                               SyclModule_t *sycl_module) {
  try {
    *sycl_module =  prtc::DynamicLibrary::open(path);
  } catch (const std::exception& e) {
    return CeedError((ceed), CEED_ERROR_BACKEND, e.what());
  }
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Compile kernel source to a shared library
// ------------------------------------------------------------------------------
int CeedBuildModule_Sycl(Ceed ceed, const std::string &kernel_source, SyclModule_t *sycl_module, const std::map<std::string, CeedInt> &constants) {
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
int CeedGetKernel_Sycl(Ceed ceed, const SyclModule_t sycl_module, const std::string &kernel_name, SyclKernel_t *sycl_kernel) {
  try {
    *sycl_kernel = sycl_module->getSymbol(kernel_name);
  } catch (const std::exception& e) {
    return CeedError((ceed), CEED_ERROR_BACKEND, e.what());
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run SYCL kernel for spatial dimension with shared memory
//------------------------------------------------------------------------------
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
