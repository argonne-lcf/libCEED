// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <map>
#include <sycl/sycl.hpp>

#include "libprtc/prtc.h"

using SyclModule_t = std::shared_ptr<prtc::DynamicLibrary>;
using SyclBundle_t = sycl::kernel_bundle<sycl::bundle_state::executable>;

int CeedBuildModule_Sycl(Ceed ceed, const std::string &kernel_source, SyclModule_t* sycl_module,
                                     const std::map<std::string, CeedInt> &constants = {});

// template <class SyclKernel_t>
// int CeedGetKernel_Sycl(Ceed ceed, const SyclModule_t sycl_module, const std::string &kernel_name, SyclKernel_t *sycl_kernel);

template <class SyclKernel_t>
int CeedGetKernel_Sycl(Ceed ceed, SyclModule_t sycl_module, std::string kernel_name, SyclKernel_t **sycl_kernel) {
  try {
    *sycl_kernel = sycl_module->getFunction<SyclKernel_t*>(kernel_name);
    // std::cout<<"\n Entered GetKernel\n";
    // void *kernel_ptr = sycl_module->getFunction2(kernel_name);
    // std::cout<<"\n Kernel pointer retrieved\n";
    // SyclKernel_t *temp = reinterpret_cast<SyclKernel_t*>(kernel_ptr);
    // std::cout<<"\n Kernel pointer recast\n";
    // sycl_kernel = temp;
  } catch (const std::exception& e) {
   return CeedError((ceed), CEED_ERROR_BACKEND, e.what());
  }
  return CEED_ERROR_SUCCESS;
}

template <class SyclKernel_t>
int CeedRunKernelDimSharedSycl(Ceed ceed, SyclKernel_t *kernel, const int grid_size, const int block_size_x, const int block_size_y,
                                           const int block_size_z, const int shared_mem_size, void **args);
