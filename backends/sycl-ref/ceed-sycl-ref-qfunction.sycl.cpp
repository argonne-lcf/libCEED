// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "../sycl/ceed-sycl-common.hpp"
#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref-qfunction-load.hpp"
#include "ceed-sycl-ref.hpp"

#define WG_SIZE_QF 384

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Sycl(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  Ceed                ceed;
  Ceed_Sycl          *ceed_Sycl;
  void               *context_data;
  CeedInt             num_input_fields, num_output_fields;
  CeedQFunction_Sycl *impl;

  CeedCallBackend(CeedQFunctionGetData(qf, &impl));

  // Build and compile kernel, if not done
  if (!impl->QFunction) CeedCallBackend(CeedQFunctionBuildKernel_Sycl(qf));

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Sycl));
  CeedCallBackend(CeedDestroy(&ceed));

  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields));

  // Read vectors
  // std::vector<const CeedScalar *> inputs(num_input_fields);
  // const CeedVector               *U_i = U;
  // for (auto &input_i : inputs) {
  //   CeedCallBackend(CeedVectorGetArrayRead(*U_i, CEED_MEM_DEVICE, &input_i));
  //   ++U_i;
  // }

  // std::vector<CeedScalar *> outputs(num_output_fields);
  // CeedVector               *V_i = V;
  // for (auto &output_i : outputs) {
  //   CeedCallBackend(CeedVectorGetArrayWrite(*V_i, CEED_MEM_DEVICE, &output_i));
  //   ++V_i;
  // }
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedVectorGetArrayRead(U[i], CEED_MEM_DEVICE, &impl->fields.inputs[i]));
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedVectorGetArrayWrite(V[i], CEED_MEM_DEVICE, &impl->fields.outputs[i]));
  }

  // Get context data
  CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &context_data));

  // Launch as a basic parallel_for over Q quadrature points
    // Hard-coding the work-group size for now
    // We could use the Level Zero API to query and set an appropriate size in future
    // Equivalent of CUDA Occupancy Calculator
  int               wg_size   = WG_SIZE_QF;
  sycl::range<1>    rounded_Q = ((Q + (wg_size - 1)) / wg_size) * wg_size;
  sycl::nd_range<1> kernel_range(rounded_Q, wg_size);

  // Call launcher function that executes kernel
  // Pass in nd_range as second argument
  // Pass in vector of events as third argument
  (*impl->QFunction)(ceed_Sycl->sycl_queue, kernel_range, context_data, Q, &impl->fields);

  // Restore vectors
  // U_i = U;
  // for (auto &input_i : inputs) {
  //   CeedCallBackend(CeedVectorRestoreArrayRead(*U_i, &input_i));
  //   ++U_i;
  // }

  // V_i = V;
  // for (auto &output_i : outputs) {
  //   CeedCallBackend(CeedVectorRestoreArray(*V_i, &output_i));
  //   ++V_i;
  // }
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U[i], &impl->fields.inputs[i]));
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedVectorRestoreArray(V[i], &impl->fields.outputs[i]));
  }

  // Restore context
  CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &context_data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Sycl(CeedQFunction qf) {
  Ceed                ceed;
  CeedQFunction_Sycl *impl;

  CeedCallBackend(CeedQFunctionGetData(qf, &impl));
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  delete impl->QFunction;
  // delete impl->sycl_module;
  CeedCallBackend(CeedFree(&impl));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Sycl(CeedQFunction qf) {
  Ceed                ceed;
  CeedQFunction_Sycl *impl;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedQFunctionSetData(qf, impl));

  // Read QFunction source
  CeedCallBackend(CeedQFunctionGetKernelName(qf, &impl->qfunction_name));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading QFunction User Source -----\n");
  CeedCallBackend(CeedQFunctionLoadSourceToBuffer(qf, &impl->qfunction_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading QFunction User Source Complete! -----\n");

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Sycl));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
