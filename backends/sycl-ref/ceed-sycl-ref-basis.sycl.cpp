// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>

#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
int CeedBasisApply_Sycl(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor
//------------------------------------------------------------------------------
int CeedBasisApplyNonTensor_Sycl(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                 CeedVector v) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Destroy tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Sycl(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Sycl *impl;
  CeedCallBackend(CeedBasisGetData(basis, &impl));
   Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());

  CeedCallSycl(ceed, sycl::free(impl->d_q_weight_1d, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_interp_1d, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_grad_1d, data->sycl_context));

  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Sycl(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasisNonTensor_Sycl *impl;
  CeedCallBackend(CeedBasisGetData(basis, &impl));
   Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());

  CeedCallSycl(ceed, sycl::free(impl->d_q_weight, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_interp, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_grad, data->sycl_context));

  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Sycl(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                 const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Sycl *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  CeedCallSycl(ceed, impl->d_q_weight_1d = sycl::malloc_device<CeedScalar>(Q_1d,data->sycl_device,data->sycl_context));
  sycl::event copy_weight = data->sycl_queue.copy<CeedScalar>(q_weight_1d, impl->d_q_weight_1d, Q_1d);

  CeedCallSycl(ceed, impl->d_interp_1d = sycl::malloc_device<CeedScalar>(P_1d,data->sycl_device,data->sycl_context));
  sycl::event copy_interp = data->sycl_queue.copy<CeedScalar>(interp_1d, impl->d_interp_1d, P_1d);

  CeedCallSycl(ceed, impl->d_grad_1d = sycl::malloc_device<CeedScalar>(P_1d,data->sycl_device,data->sycl_context));
  sycl::event copy_grad = data->sycl_queue.copy<CeedScalar>(grad_1d, impl->d_grad_1d, P_1d);

  CeedCallSycl(ceed, sycl::event::wait_and_throw({copy_weight,copy_interp,copy_grad}));

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Apply", CeedBasisApply_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Sycl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Sycl(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                           const CeedScalar *qref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasisNonTensor_Sycl *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  CeedCallSycl(ceed, impl->d_q_weight = sycl::malloc_device<CeedScalar>(num_qpts,data->sycl_device,data->sycl_context));
  sycl::event copy_weight = data->sycl_queue.copy<CeedScalar>(q_weight, impl->d_q_weight, num_qpts);

  CeedCallSycl(ceed, impl->d_interp = sycl::malloc_device<CeedScalar>(num_nodes,data->sycl_device,data->sycl_context));
  sycl::event copy_interp = data->sycl_queue.copy<CeedScalar>(interp, impl->d_interp, num_nodes);

  const CeedInt grad_length = num_nodes * dim;
  CeedCallSycl(ceed, impl->d_grad = sycl::malloc_device<CeedScalar>(grad_length,data->sycl_device,data->sycl_context));
  sycl::event copy_grad = data->sycl_queue.copy<CeedScalar>(grad, impl->d_grad, grad_length);

  CeedCallSycl(ceed, sycl::event::wait_and_throw({copy_weight,copy_interp,copy_grad}));

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Sycl));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
