// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#define CEED_DEBUG_COLOR 12

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "../sycl-ref/ceed-sycl-ref.hpp"
#include "../sycl-shared/ceed-sycl-shared.hpp"
#include "../sycl/ceed-sycl-compile.hpp"

#include "ceed-sycl-gen.hpp"

//------------------------------------------------------------------------------
// Calculate the block size used for launching the operator kernel
//------------------------------------------------------------------------------
extern "C" int BlockGridCalculate_Sycl_gen(const CeedInt dim, const CeedInt P_1d, const CeedInt Q_1d, CeedInt *block_sizes) {
  const CeedInt thread1d = CeedIntMax(Q_1d, P_1d);
  if (dim == 1) {
    CeedInt elems_per_block = 64 * thread1d > 256 ? 256 / thread1d : 64;
    elems_per_block         = elems_per_block > 0 ? elems_per_block : 1;
    block_sizes[0]          = thread1d;
    block_sizes[1]          = 1;
    block_sizes[2]          = elems_per_block;
  } else if (dim == 2) {
    const CeedInt elems_per_block = thread1d < 4 ? 16 : 2;
    block_sizes[0]                = thread1d;
    block_sizes[1]                = thread1d;
    block_sizes[2]                = elems_per_block;
  } else if (dim == 3) {
    const CeedInt elems_per_block = thread1d < 6 ? 4 : (thread1d < 8 ? 2 : 1);
    block_sizes[0]                = thread1d;
    block_sizes[1]                = thread1d;
    block_sizes[2]                = elems_per_block;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Build single operator kernel
//------------------------------------------------------------------------------
extern "C" int CeedSyclGenOperatorBuild(CeedOperator op) {
  bool is_setup_done;
  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;
 
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  Ceed_Sycl *sycl_data;
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  CeedOperator_Sycl_gen *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedQFunction          qf;
  CeedQFunction_Sycl_gen *qf_impl;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_impl));
  CeedSize lsize;
  CeedInt  Q, P_1d = 0, Q_1d = 0, elem_size, num_input_fields, num_output_fields, num_comp, dim = 1;
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  Q_1d = Q;

  CeedOperatorField *op_input_fields, *op_output_fields;
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  
  CeedEvalMode             eval_mode;
  CeedBasis                basis;
  CeedBasis_Sycl_shared    *basis_impl;
  CeedElemRestriction      Erestrict;
  CeedElemRestriction_Sycl *restr_impl;

  // Check for restriction only identity operator
  bool is_identity_qf;
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &is_identity_qf));
  if (is_identity_qf) {
    CeedEvalMode eval_mode_in, eval_mode_out;
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[0], &eval_mode_in));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[0], &eval_mode_out));
    if (eval_mode_in == CEED_EVAL_NONE && eval_mode_out == CEED_EVAL_NONE) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement restriction only identity operators");
      // LCOV_EXCL_STOP
    }
  }

  std::ostringstream code;
  // TODO: generalize to accept different device functions?
  {
    char *tensor_basis_kernel_path, *tensor_basis_kernel_source;
    CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/sycl/sycl-shared-basis-tensor-templates.h", &tensor_basis_kernel_path));
    CeedDebug256(ceed, 2, "----- Loading Tensor Basis Kernel Source -----\n");
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, tensor_basis_kernel_path, &tensor_basis_kernel_source));
    code << tensor_basis_kernel_source;
    CeedCallBackend(CeedFree(&tensor_basis_kernel_path));
    CeedCallBackend(CeedFree(&tensor_basis_kernel_source));
  }
  {
    char *sycl_gen_template_path, *sycl_gen_template_source;
    CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/sycl/sycl-gen-templates.h", &sycl_gen_template_path));
    CeedDebug256(ceed, 2, "----- Loading Sycl-Gen Template Source -----\n");
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, sycl_gen_template_path, &sycl_gen_template_source));
    code << sycl_gen_template_source;
    CeedCallBackend(CeedFree(&sycl_gen_template_path));
    CeedCallBackend(CeedFree(&sycl_gen_template_source));
  }

  std::string_view q_function_source(qf_impl->q_function_source);
  std::string_view q_function_name(qf_impl->q_function_name);
  const std::string operator_name = "CeedKernelSyclGenOperator_" + std::string(q_function_name);

  // Find dim, P_1d, Q_1d
  impl->max_P_1d = 0;
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
    if (basis != CEED_BASIS_COLLOCATED) {
      CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));

      // Collect dim, P_1d, and Q_1d
      CeedCallBackend(CeedBasisGetDimension(basis, &dim));
      bool isTensor;
      CeedCallBackend(CeedBasisIsTensor(basis, &isTensor));
      if (isTensor) {
        CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
        CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
        if (P_1d > impl->max_P_1d) impl->max_P_1d = P_1d;
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
        // LCOV_EXCL_STOP
      }
    }
  }
  // Check output bases for Q_1d, dim as well
  //   The only input basis might be CEED_BASIS_COLLOCATED
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));

    if (basis != CEED_BASIS_COLLOCATED) {
      CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));

      // Collect Q_1d
      CeedCallBackend(CeedBasisGetDimension(basis, &dim));
      bool isTensor;
      CeedCallBackend(CeedBasisIsTensor(basis, &isTensor));
      if (isTensor) {
        CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
        // LCOV_EXCL_STOP
      }
    }
  }
  impl->dim  = dim;
  impl->Q_1d = Q_1d;

  // Only use 3D collocated gradient parallelization strategy when gradient is computed
  // TODO: put in a function?
  bool use_collograd_parallelization = false;
  if (dim == 3) {
    bool was_grad_found = false;
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_GRAD) {
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        use_collograd_parallelization = !!basis_impl->d_collo_grad_1d && (was_grad_found ? use_collograd_parallelization : true);
        was_grad_found                = true;
      }
    }
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_GRAD) {
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        use_collograd_parallelization = !!basis_impl->d_collo_grad_1d && (was_grad_found ? use_collograd_parallelization : true);
        was_grad_found                = true;
      }
    }
  }

  // Define CEED_Q_VLA
  code << "\n#undef CEED_Q_VLA\n";
  if (dim != 3 || use_collograd_parallelization) {
    code << "#define CEED_Q_VLA 1\n\n";
  } else {
    code << "#define CEED_Q_VLA " << Q_1d << "\n\n";
  }

  code << q_function_source;

  // Setup
  code << "\n// -----------------------------------------------------------------------------\n";
  // code << "\nextern \"C\" __launch_bounds__(BLOCK_SIZE)\n";
  // TODO: Need to convert the input structs like in sycl-ref
  code << "kernel void " << operator_name << "(CeedInt num_elem, void* ctx, FieldsInt_Sycl indices, Fields_Sycl fields, Fields_Sycl B, Fields_Sycl G, CeedScalar* W) {\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {  // Skip CEED_EVAL_WEIGHT
      code << "  const CeedScalar* d_u_" << i << " = fields.inputs[" << i << "];\n";
    }
  }

  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "  CeedScalar* d_v_" << i << " = fields.outputs[" << i << "];\n";
  }

  code << "  const CeedInt dim = " << dim << ";\n";
  code << "  const CeedInt Q_1d = " << Q_1d << ";\n";

  const CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
  CeedInt       block_sizes[3];
  CeedCallBackend(BlockGridCalculate_Sycl_gen(dim, P_1d, Q_1d, block_sizes));
  const CeedInt scratch_size = block_sizes[2] * thread_1d * ((dim > 1) ? thread_1d : 1);
  code << "  local CeedScalar scratch[" << scratch_size << "];\n";
  code << "  local CeedScalar * elem_scratch = scratch + get_local_id(2) * T_1D" << (dim > 1 ? "*T_1D" : "") << ";\n";

  code << "\n  // -- Input field constants and basis data --\n";
  // Initialize constants, and matrices B and G
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "  // ---- Input field " << i << " ----\n";
    // Get elem_size, eval_mode, num_comp
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &Erestrict));
    CeedCallBackend(CeedElemRestrictionGetElementSize(Erestrict, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(Erestrict, &num_comp));

    // Set field constants
    if (eval_mode != CEED_EVAL_WEIGHT) {
      CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
      if (basis != CEED_BASIS_COLLOCATED) {
        CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
        code << "  const CeedInt P_in_" << i << " = " << P_1d << ";\n";
      } else {
        code << "  const CeedInt P_in_" << i << " = " << Q_1d << ";\n";
      }
      code << "  const CeedInt num_comp_in_" << i << " = " << num_comp << ";\n";
    }

    // Load basis data
    code << "  // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        impl->B.inputs[i] = basis_impl->d_interp_1d;
        code << "  local CeedScalar s_B_in_" << i << "[" << P_1d * Q_1d << "];\n";
        code << "  loadMatrix(P_in_" << i << "*Q_1d, B.inputs[" << i << "], s_B_in_" << i << ");\n";
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        impl->B.inputs[i] = basis_impl->d_interp_1d;
        code << "  local CeedScalar s_B_in_" << i << "[" << P_1d * Q_1d << "];\n";
        code << "  loadMatrix(P_in_" << i << "*Q_1d, B.inputs[" << i << "], s_B_in_" << i << ");\n";
        if (use_collograd_parallelization) {
          impl->G.inputs[i] = basis_impl->d_collo_grad_1d;
          code << "  local CeedScalar s_G_in_" << i << "[" << Q_1d * Q_1d << "];\n";
          code << "  loadMatrix(Q_1d*Q_1d, G.inputs[" << i << "], s_G_in_" << i << ");\n";
        } else {
          bool has_collo_grad = !!basis_impl->d_collo_grad_1d;
          impl->G.inputs[i]   = has_collo_grad ? basis_impl->d_collo_grad_1d : basis_impl->d_grad_1d;
          code << "  local CeedScalar s_G_in_" << i << "[" << Q_1d * (has_collo_grad ? Q_1d : P_1d) << "];\n";
          code << "  loadMatrix(" << (has_collo_grad ? "Q_1d" : ("P_in_" + std::to_string(i))) << "*Q_1d, G.inputs[" << i << "], s_G_in_" << i
               << ");\n";
        }
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
    }
  }

  code << "\n  // -- Output field constants and basis data --\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "  // ---- Output field " << i << " ----\n";
    // Get elem_size, eval_mode, num_comp
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &Erestrict));
    CeedCallBackend(CeedElemRestrictionGetElementSize(Erestrict, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(Erestrict, &num_comp));

    // Set field constants
    CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
    if (basis != CEED_BASIS_COLLOCATED) {
      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      code << "  const CeedInt P_out_" << i << " = " << P_1d << ";\n";
    } else {
      code << "  const CeedInt P_out_" << i << " = " << Q_1d << ";\n";
    }
    code << "  const CeedInt num_comp_out_" << i << " = " << num_comp << ";\n";

    // Load basis data
    code << "  // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        impl->B.outputs[i] = basis_impl->d_interp_1d;
        code << "  local CeedScalar s_B_out_" << i << "[" << P_1d * Q_1d << "];\n";
        code << "  loadMatrix(P_out_" << i << "*Q_1d, B.outputs[" << i << "], s_B_out_" << i << ");\n";
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        impl->B.outputs[i] = basis_impl->d_interp_1d;
        code << "  local CeedScalar s_B_out_" << i << "[" << P_1d * Q_1d << "];\n";
        code << "  loadMatrix(P_out_" << i << "*Q_1d, B.outputs[" << i << "], s_B_out_" << i << ");\n";
        if (use_collograd_parallelization) {
          impl->G.outputs[i] = basis_impl->d_collo_grad_1d;
          code << "  local CeedScalar s_G_out_" << i << "[" << Q_1d * Q_1d << "];\n";
          code << "  loadMatrix(Q_1d*Q_1d, G.outputs[" << i << "], s_G_out_" << i << ");\n";
        } else {
          bool has_collo_grad = !!basis_impl->d_collo_grad_1d;
          impl->G.outputs[i]  = has_collo_grad ? basis_impl->d_collo_grad_1d : basis_impl->d_grad_1d;
          code << "  local CeedScalar s_G_out_" << i << "[" << Q_1d * (has_collo_grad ? Q_1d : P_1d) << "];\n";
          code << "  loadMatrix(" << (has_collo_grad ? "Q_1d" : ("P_out_" + std::to_string(i))) << "*Q_1d, G.outputs[" << i << "], s_G_out_"
               << i << ");\n";
        }
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        Ceed ceed;
        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break;  // Should not occur
      }
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
                // LCOV_EXCL_STOP
    }
  }
  code << "\n  // -- Element loop --\n";
  code << "  work_group_barrier(CLK_LOCAL_MEM_FENCE);\n";
  code << "  for (CeedInt elem = get_local_size(0)*get_local_size(2) + get_local_id(2); elem < num_elem; elem += get_group_id(0)*get_local_size(2)) {\n";
  // Input basis apply if needed
  // Generate the correct eval mode code for each input
  code << "    // -- Input field restrictions and basis actions --\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "    // ---- Input field " << i << " ----\n";
    // Get elem_size, eval_mode, num_comp
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &Erestrict));
    CeedCallBackend(CeedElemRestrictionGetElementSize(Erestrict, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(Erestrict, &num_comp));

    // Restriction
    if (eval_mode != CEED_EVAL_WEIGHT && !((eval_mode == CEED_EVAL_NONE) && use_collograd_parallelization)) {
      code << "    CeedScalar r_u_" << i << "[num_comp_in_" << i << "*P_in_" << i << "];\n";

      bool is_strided;
      CeedCallBackend(CeedElemRestrictionIsStrided(Erestrict, &is_strided));
      if (!is_strided) {
        CeedCallBackend(CeedElemRestrictionGetLVectorSize(Erestrict, &lsize));
        code << "    const CeedInt lsize_in_" << i << " = " << lsize << ";\n";
        CeedInt comp_stride;
        CeedCallBackend(CeedElemRestrictionGetCompStride(Erestrict, &comp_stride));
        code << "    // CompStride: " << comp_stride << "\n";
        CeedCallBackend(CeedElemRestrictionGetData(Erestrict, &restr_impl));
        impl->indices.inputs[i] = restr_impl->d_ind;
        code << "    readDofsOffset" << dim << "d(num_comp_in_" << i << ", " << comp_stride << ", P_in_" << i << ", lsize_in_" << i
             << ", elem, indices.inputs[" << i << "], d_u_" << i << ", r_u_" << i << ");\n";
      } else {
        bool has_backend_strides;
        CeedCallBackend(CeedElemRestrictionHasBackendStrides(Erestrict, &has_backend_strides));
        CeedInt num_elem;
        CeedCallBackend(CeedElemRestrictionGetNumElements(Erestrict, &num_elem));
        CeedInt strides[3] = {1, elem_size * num_elem, elem_size};
        if (!has_backend_strides) {
          CeedCallBackend(CeedElemRestrictionGetStrides(Erestrict, &strides));
        }
        code << "    // Strides: {" << strides[0] << ", " << strides[1] << ", " << strides[2] << "}\n";
        code << "    readDofsStrided" << dim << "d(num_comp_in_" << i << ",P_in_" << i << "," << strides[0] << "," << strides[1] << "," << strides[2]
             << ", elem, d_u_" << i << ", r_u_" << i << ");\n";
      }
    }

    // Basis action
    code << "    // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        if (!use_collograd_parallelization) {
          code << "    CeedScalar* r_t_" << i << " = r_u_" << i << ";\n";
        }
        break;
      case CEED_EVAL_INTERP:
        code << "    CeedScalar r_t_" << i << "[num_comp_in_" << i << "*Q_1d];\n";
        code << "    Interp" << (dim > 1 ? "Tensor" : "") << dim << "d(num_comp_in_" << i << ", P_in_" << i << ", Q_1d, r_u_" << i << ", s_B_in_"
             << i << ", r_t_" << i << ");\n";
        break;
      case CEED_EVAL_GRAD:
        if (use_collograd_parallelization) {
          code << "    CeedScalar r_t_" << i << "[num_comp_in_" << i << "*Q_1d];\n";
          code << "    Interp" << (dim > 1 ? "Tensor" : "") << dim << "d(num_comp_in_" << i << ", P_in_" << i << ", Q_1d, r_u_" << i
               << ", s_B_in_" << i << ", r_t_" << i << ");\n";
        } else {
          CeedInt P_1d;
          CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
          CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
          code << "    CeedScalar r_t_" << i << "[num_comp_in_" << i << "*dim*Q_1d];\n";
          code << "    Grad" << (dim > 1 ? "Tensor" : "") << (dim == 3 && Q_1d >= P_1d ? "Collocated" : "") << dim << "d(num_comp_in_" << i
               << ", P_in_" << i << ", Q_1d, r_u_" << i << ", s_B_in_" << i << ", s_G_in_" << i << ", r_t_" << i << ");\n";
        }
        break;
      case CEED_EVAL_WEIGHT:
        code << "    CeedScalar r_t_" << i << "[Q_1d];\n";
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        impl->W = basis_impl->d_q_weight_1d;
        code << "    Weight" << (dim > 1 ? "Tensor" : "") << dim << "d(Q_1d, W, r_t_" << i << ");\n";
        break;  // No action
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
    }
  }

  // Q function
  code << "\n    // -- Output field setup --\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "\n    // ---- Output field " << i << " ----\n";
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_GRAD) {
      if (use_collograd_parallelization) {
        // Accumulator for gradient slices
        code << "    CeedScalar r_tt_" << i << "[num_comp_out_" << i << "*Q_1d];\n";
        code << "    for (CeedInt i = 0; i < num_comp_out_" << i << "; i++) {\n";
        code << "      for (CeedInt j = 0; j < Q_1d; ++j) {\n";
        code << "        r_tt_" << i << "[j + i*Q_1d] = 0.0;\n";
        code << "      }\n";
        code << "    }\n";
      } else {
        code << "    CeedScalar r_tt_" << i << "[num_comp_out_" << i << "*dim*Q_1d];\n";
      }
    }
    if (eval_mode == CEED_EVAL_NONE || eval_mode == CEED_EVAL_INTERP) {
      code << "    CeedScalar r_tt_" << i << "[num_comp_out_" << i << "*Q_1d];\n";
    }
  }
  // We treat quadrature points per slice in 3d to save registers
  if (use_collograd_parallelization) {
    code << "\n    // Note: Using planes of 3D elements\n";
    code << "#pragma unroll\n"; // TODO: check syntax for OpenCL unrolling
    code << "    for (CeedInt q = 0; q < Q_1d; q++) {\n";
    code << "      // -- Input fields --\n";
    for (CeedInt i = 0; i < num_input_fields; i++) {
      code << "      // ---- Input field " << i << " ----\n";
      // Get elem_size, eval_mode, num_comp
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      // Basis action
      code << "      // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      CeedScalar r_q_" << i << "[num_comp_in_" << i << "];\n";

          bool is_strided;
          CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &Erestrict));
          CeedCallBackend(CeedElemRestrictionIsStrided(Erestrict, &is_strided));
          if (!is_strided) {
            CeedCallBackend(CeedElemRestrictionGetLVectorSize(Erestrict, &lsize));
            code << "      const CeedInt lsize_in_" << i << " = " << lsize << ";\n";
            CeedInt comp_stride;
            CeedCallBackend(CeedElemRestrictionGetCompStride(Erestrict, &comp_stride));
            code << "      // CompStride: " << comp_stride << "\n";
            CeedCallBackend(CeedElemRestrictionGetData(Erestrict, &restr_impl));
            impl->indices.inputs[i] = restr_impl->d_ind;
            code << "      readSliceQuadsOffset"
                 << "3d(num_comp_in_" << i << ", " << comp_stride << ", Q_1d, lsize_in_" << i << ", elem, q, indices.inputs[" << i << "], d_u_"
                 << i << ", r_q_" << i << ");\n";
          } else {
            CeedCallBackend(CeedElemRestrictionGetElementSize(Erestrict, &elem_size));
            bool has_backend_strides;
            CeedCallBackend(CeedElemRestrictionHasBackendStrides(Erestrict, &has_backend_strides));
            CeedInt num_elem;
            CeedCallBackend(CeedElemRestrictionGetNumElements(Erestrict, &num_elem));
            CeedInt strides[3] = {1, elem_size * num_elem, elem_size};
            if (!has_backend_strides) {
              CeedCallBackend(CeedElemRestrictionGetStrides(Erestrict, &strides));
            }
            code << "      // Strides: {" << strides[0] << ", " << strides[1] << ", " << strides[2] << "}\n";
            code << "      readSliceQuadsStrided"
                 << "3d(num_comp_in_" << i
                 << ", Q_1d,"
                 << strides[0] << ", " << strides[1] << ", " << strides[2] << ", elem, q, d_u_" << i << ", r_q_" << i << ");\n";
          }
          break;
        case CEED_EVAL_INTERP:
          code << "      CeedScalar r_q_" << i << "[num_comp_in_" << i << "];\n";
          code << "      for (CeedInt j = 0; j < num_comp_in_" << i << " ; ++j) {\n";
          code << "        r_q_" << i << "[j] = r_t_" << i << "[q + j*Q_1d];\n";
          code << "      }\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      CeedScalar r_q_" << i << "[num_comp_in_" << i << "*dim];\n";
          code << "      gradCollo3d(num_comp_in_" << i << ", Q_1d, q, r_t_" << i << ", s_G_in_" << i << ", r_q_" << i << ");\n";
          break;
        case CEED_EVAL_WEIGHT:
          code << "      CeedScalar r_q_" << i << "[1];\n";
          code << "      r_q_" << i << "[0] = r_t_" << i << "[q];\n";
          break;  // No action
        case CEED_EVAL_DIV:
          break;  // TODO: Not implemented
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
      }
    }
    code << "\n      // -- Output fields --\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      code << "      // ---- Output field " << i << " ----\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      CeedScalar r_qq_" << i << "[num_comp_out_" << i << "];\n";
          break;  // No action
        case CEED_EVAL_INTERP:
          code << "      CeedScalar r_qq_" << i << "[num_comp_out_" << i << "];\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      CeedScalar r_qq_" << i << "[num_comp_out_" << i << "*dim];\n";
          break;
        case CEED_EVAL_WEIGHT:
          break;  // Should not occur
        case CEED_EVAL_DIV:
          break;  // TODO: Not implemented
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
      }
    }
  } else {
    code << "\n      // Note: Using full elements\n";
    code << "      // -- Input fields --\n";
    for (CeedInt i = 0; i < num_input_fields; i++) {
      code << "      // ---- Input field " << i << " ----\n";
      code << "      CeedScalar* r_q_" << i << " = r_t_" << i << ";\n";
    }
    code << "      // -- Output fields --\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      code << "      // ---- Output field " << i << " ----\n";
      code << "      CeedScalar* r_qq_" << i << " = r_tt_" << i << ";\n";
    }
  }
  code << "\n      // -- QFunction Inputs and outputs --\n";
  code << "      CeedScalar* in[" << num_input_fields << "];\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "      // ---- Input field " << i << " ----\n";
    code << "      in[" << i << "] = r_q_" << i << ";\n";
  }
  code << "      CeedScalar* out[" << num_output_fields << "];\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "      // ---- Output field " << i << " ----\n";
    code << "      out[" << i << "] = r_qq_" << i << ";\n";
  }
  code << "\n      // -- Apply QFunction --\n";
  code << "      " << q_function_name << "(ctx, ";
  if (dim != 3 || use_collograd_parallelization) {
    code << "1";
  } else {
    code << "Q_1d";
  }
  code << ", in, out);\n";
  if (use_collograd_parallelization) {
    code << "      // -- Output fields --\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      code << "      // ---- Output field " << i << " ----\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      code << "      // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      for (CeedInt j = 0; j < num_comp_out_" << i << " ; ++j) {\n";
          code << "        r_tt_" << i << "[q + j*Q_1d] = r_qq_" << i << "[j];\n";
          code << "      }\n";
          break;  // No action
        case CEED_EVAL_INTERP:
          code << "      for (CeedInt j = 0; j < num_comp_out_" << i << " ; ++j) {\n";
          code << "        r_tt_" << i << "[q + j*Q_1d] = r_qq_" << i << "[j];\n";
          code << "      }\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      gradColloTranspose3d(num_comp_out_" << i << ",Q_1d, q, r_qq_" << i << ", s_G_out_" << i << ", r_tt_" << i << ");\n";
          break;
        case CEED_EVAL_WEIGHT:
          break;  // Should not occur
        case CEED_EVAL_DIV:
          break;  // TODO: Not implemented
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
      }
    }
    code << "    }\n";
  }

  // Output basis apply if needed
  // Generate the correct eval mode code for each output
  code << "\n    // -- Output field basis action and restrictions --\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "    // ---- Output field " << i << " ----\n";
    // Get elem_size, eval_mode, num_comp
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &Erestrict));
    CeedCallBackend(CeedElemRestrictionGetElementSize(Erestrict, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(Erestrict, &num_comp));
    // Basis action
    code << "    // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        code << "    CeedScalar* r_v_" << i << " = r_tt_" << i << ";\n";
        break;  // No action
      case CEED_EVAL_INTERP:
        code << "    CeedScalar r_v_" << i << "[num_comp_out_" << i << "*P_out_" << i << "];\n";
        code << "    InterpTranspose" << (dim > 1 ? "Tensor" : "") << dim << "d(num_comp_out_" << i << ",P_out_" << i << ", Q_1d, r_tt_" << i
             << ", s_B_out_" << i << ", r_v_" << i << ");\n";
        break;
      case CEED_EVAL_GRAD:
        code << "    CeedScalar r_v_" << i << "[num_comp_out_" << i << "*P_out_" << i << "];\n";
        if (use_collograd_parallelization) {
          code << "    InterpTranspose" << (dim > 1 ? "Tensor" : "") << dim << "d(num_comp_out_" << i << ",P_out_" << i << ", Q_1d, r_tt_" << i
               << ", s_B_out_" << i << ", r_v_" << i << ");\n";
        } else {
          CeedInt P_1d;
          CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
          CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
          code << "    GradTranspose" << (dim > 1 ? "Tensor" : "") << (dim == 3 && Q_1d >= P_1d ? "Collocated" : "") << dim << "d(num_comp_out_" << i
               << ",P_out_" << i << ", Q_1d, r_tt_" << i << ", s_B_out_" << i << ", s_G_out_" << i << ", r_v_" << i << ");\n";
        }
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        Ceed ceed;
        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break;  // Should not occur
      }
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
                // LCOV_EXCL_STOP
    }
    // Restriction
    bool is_strided;
    CeedCallBackend(CeedElemRestrictionIsStrided(Erestrict, &is_strided));
    if (!is_strided) {
      CeedCallBackend(CeedElemRestrictionGetLVectorSize(Erestrict, &lsize));
      code << "    const CeedInt lsize_out_" << i << " = " << lsize << ";\n";
      CeedInt comp_stride;
      CeedCallBackend(CeedElemRestrictionGetCompStride(Erestrict, &comp_stride));
      code << "    // CompStride: " << comp_stride << "\n";
      CeedCallBackend(CeedElemRestrictionGetData(Erestrict, &restr_impl));
      impl->indices.outputs[i] = restr_impl->d_ind;
      code << "    writeDofsOffset" << dim << "d(num_comp_out_" << i << ", " << comp_stride << ", P_out_" << i << ", lsize_out_" << i
           << ", elem, indices.outputs[" << i << "], r_v_" << i << ", d_v_" << i << ");\n";
    } else {
      bool has_backend_strides;
      CeedCallBackend(CeedElemRestrictionHasBackendStrides(Erestrict, &has_backend_strides));
      CeedInt num_elem;
      CeedCallBackend(CeedElemRestrictionGetNumElements(Erestrict, &num_elem));
      CeedInt strides[3] = {1, elem_size * num_elem, elem_size};
      if (!has_backend_strides) {
        CeedCallBackend(CeedElemRestrictionGetStrides(Erestrict, &strides));
      }
      code << "    // Strides: {" << strides[0] << ", " << strides[1] << ", " << strides[2] << "}\n";
      code << "    writeDofsStrided" << dim << "d(num_comp_out_" << i << ",P_out_" << i << "," << strides[0] << "," << strides[1] << "," << strides[2]
           << ", elem, r_v_" << i << ", d_v_" << i << ");\n";
    }
  }

  code << "  }\n";
  code << "}\n";
  code << "// -----------------------------------------------------------------------------\n\n";

  std::cout << code.str() << std::endl;
  // View kernel for debugging
  CeedDebug256(ceed, 2, "Generated Operator Kernels:\n");
  CeedDebug(ceed, code.str().c_str());

  // CeedInt block_sizes[3] = {0, 0, 0};
  // CeedInt num_elem;
  // CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  // CeedCallBackend(BlockGridCalculate_Sycl_gen(dim, num_elem, impl->max_P_1d, Q_1d, block_sizes));
  // CeedCallBackend(CeedCompileSycl(ceed, code.str().c_str(), &impl->module, 2, "T_1D", block_sizes[0], "BLOCK_SIZE",
  //                                block_sizes[0] * block_sizes[1] * block_sizes[2]));
  // CeedCallBackend(CeedGetKernelSycl(ceed, impl->module, operator_name.c_str(), &impl->op));

  // CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
