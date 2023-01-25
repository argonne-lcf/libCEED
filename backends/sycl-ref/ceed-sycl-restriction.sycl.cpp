// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>

#include <string>
#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Sycl(CeedElemRestriction r, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionApplyBlock_Sycl(CeedElemRestriction r, CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                       CeedRequest *request) {
  // LCOV_EXCL_START
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Sycl(CeedElemRestriction r, CeedMemType m_type, const CeedInt **offsets) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Destroy restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Sycl(CeedElemRestriction r) {
  CeedElemRestriction_Sycl *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());
  CeedCallSycl(ceed, sycl::free(impl->d_ind_allocated , data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_t_offsets     , data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_t_indices     , data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_l_vec_indices , data->sycl_context));
  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create transpose offsets and indices
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffset_Sycl(const CeedElemRestriction r, const CeedInt *indices) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Create restriction
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Sycl(CeedMemType m_type, CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionCreateBlocked_Sycl(const CeedMemType m_type, const CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement blocked restrictions");
}
//------------------------------------------------------------------------------
