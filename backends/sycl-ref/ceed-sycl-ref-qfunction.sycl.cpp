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

class CeedQFunction_setup;

extern "C" void CeedKernelSyclRefQFunction_setup(sycl::queue &sycl_queue, sycl::nd_range<1> kernel_range, void *ctx, CeedInt Q, Fields_Sycl *fields) {
  const CeedScalar *fields_inputs[2];
  fields_inputs[0] = fields->inputs[0];
  fields_inputs[1] = fields->inputs[1];
  CeedScalar *fields_outputs[1];
  fields_outputs[0] = fields->outputs[0];

  std::vector<sycl::event> e;
  if (!sycl_queue.is_in_order()) e = {sycl_queue.ext_oneapi_submit_barrier()};

  sycl_queue.parallel_for<CeedQFunction_setup>(kernel_range, e, [=](sycl::nd_item<1> item) {
    // Input fields
    CeedScalar U_0[1];
    CeedScalar U_1[1];
    const CeedScalar *inputs[2] = {U_0, U_1};

    // Output fields
    CeedScalar V_0[1];
    CeedScalar *outputs[1] = {V_0};

    const CeedInt q = item.get_global_linear_id();

    if(q < Q) { 

      // -- Load inputs
      // readQuads<1>(q, Q, fields_inputs[0], U_0);
      // readQuads<1>(q, Q, fields_inputs[1], U_1);

      // -- Call QFunction
      // setup(ctx, 1, inputs, outputs);

      // -- Write outputs
      // writeQuads<1>(q, Q, V_0, fields_outputs[0]);
    }
  });
}

#include "../sycl/libprtc/prtc.h"
static int CeedLoadModule_Sycl(Ceed ceed, const sycl::context &sycl_context, const sycl::device &sycl_device, const std::string& path,
                               SyclModule_t* sycl_module) {
  try {
    *sycl_module =  prtc::DynamicLibrary::open(path);
    // std::string check_path = (*sycl_module)->path();
    // std::cout<<"\n Module loaded from path"<<check_path<<std::endl;
  } catch (const std::exception& e) {
    return CeedError((ceed), CEED_ERROR_BACKEND, e.what());
  }
  return CEED_ERROR_SUCCESS;
}

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
  // std::cout << " Number of input fields = "<<num_input_fields<<"\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedVectorGetArrayRead(U[i], CEED_MEM_DEVICE, &impl->fields.inputs[i]));
  }
  // std::cout << " Number of output fields = "<<num_output_fields<<"\n";
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
  /*
  std::string module_path = (impl->sycl_module)->path();
  std::cout<<"\n Loading Module from library";
  SyclModule_t my_module =  prtc::DynamicLibrary::open(module_path);
  // CeedLoadModule_Sycl(ceed, ceed_Sycl->sycl_context, ceed_Sycl->sycl_device, module_path, &impl->sycl_module);
  // std::string module_path = (impl->sycl_module)->path();
  std::cout<<"\n Module loaded from path"<<module_path<<std::endl;
  std::string_view  qf_name_view(impl->qfunction_name);
  const std::string kernel_name = "CeedKernelSyclRefQFunction_" + std::string(qf_name_view);
  std::cout<<"\n Loading Kernel "<<kernel_name<<" from module";
  void *kernel_ptr = my_module->getFunction2(kernel_name);
  std::cout<<"\n Kernel pointer retrieved";    
  // SyclQFunctionKernel_t *my_QFunction = new SyclQFunctionKernel_t(1);
  SyclQFunctionKernel_t *my_QFunction = reinterpret_cast<SyclQFunctionKernel_t*>(kernel_ptr);
  std::cout<<"\n Kernel pointer recast\n";
  */
  // CeedGetKernel_Sycl<SyclQFunctionKernel_t>(ceed, impl->sycl_module, kernel_name, &impl->QFunction);
  // std::cout<<"\n Launching QFunction kernel\n";
  // std::function<int(int)> *test_fun;
  if(impl->QFunction==NULL) {
    return CeedError((ceed), CEED_ERROR_BACKEND, "Kernel function is NULL\n");
  }
  // Fields_Sycl fields = impl->fields;
  // if(!fields.inputs) {
  //   return CeedError((ceed), CEED_ERROR_BACKEND, "Impl fields is NULL\n");
  // }
  // std::cout<<" QFunction pointer being deleted\n";
  // delete impl->QFunction;
  // CeedKernelSyclRefQFunction_setup(ceed_Sycl->sycl_queue, kernel_range, context_data, Q, &impl->fields);
  (*impl->QFunction)(ceed_Sycl->sycl_queue, kernel_range, context_data, Q, &impl->fields);
  // (*my_QFunction)(ceed_Sycl->sycl_queue, kernel_range, context_data, Q, &impl->fields);
  // std::cout<<" QFunction kernel successful\n";

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
  // delete impl->QFunction;
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
