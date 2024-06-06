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

#include <libprtc/prtc.h>

using SyclModule_t = std::shared_ptr<prtc::DynamicLibrary>;
using SyclKernel_t = void*; // Revisit this

CEED_INTERN int CeedBuildModule_Sycl(Ceed ceed, const std::string &kernel_source, SyclModule_t *sycl_module,
                                     const std::map<std::string, CeedInt> &constants = {});
CEED_INTERN int CeedGetKernel_Sycl(Ceed ceed, const SyclModule_t sycl_module, const std::string &kernel_name, SyclKernel_t *sycl_kernel);

CEED_INTERN int CeedRunKernelDimSharedSycl(Ceed ceed, SyclKernel_t *kernel, const int grid_size, const int block_size_x, const int block_size_y,
                                           const int block_size_z, const int shared_mem_size, void **args);
