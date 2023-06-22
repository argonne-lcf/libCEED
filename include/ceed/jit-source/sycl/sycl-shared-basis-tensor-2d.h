// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory tensor product basis
#ifndef _ceed_sycl_shared_basis_tensor_2d_h
#define _ceed_sycl_shared_basis_tensor_2d_h

#include <ceed.h>

#include "sycl-shared-basis-read-write-templates-2d.h"
#include "sycl-shared-basis-tensor-templates-2d.h"

//
// BASIS_NUM_NODES = CeedIntPow(BASIS_P_1D,2)
// BASIS_NUM_QPTS = CeedIntPow(BASIS_Q_1D,2)


//------------------------------------------------------------------------------
// Interp kernel by dim
//------------------------------------------------------------------------------
kernel void Interp(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_U, 
  global CeedScalar * restrict d_V) {

  local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  private CeedScalar r_U[BASIS_NUM_COMP];
  private CeedScalar r_V[BASIS_NUM_COMP];
  
  local CeedScalar scratch[BASIS_INTERP_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * T_1D * T_1D;

  loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  ReadElementStrided2d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES *num_elem, BASIS_NUM_NODES, d_U, r_U);
  InterpTensor2d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
  WriteElementStrided2d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS *num_elem, BASIS_NUM_QPTS, r_V, d_V);
}

kernel void InterpTranspose(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_U,
  global CeedScalar * restrict d_V) {
  // local size: 
  // 1d: elems_per_block * T_1d
  // 2d,3d: elems_per_block * T_1d * T_1d 
  local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  private CeedScalar r_U[BASIS_NUM_COMP];
  private CeedScalar r_V[BASIS_NUM_COMP];
  
  local CeedScalar scratch[BASIS_INTERP_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * T_1D * T_1D;

  loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  ReadElementStrided2d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS* num_elem, BASIS_NUM_QPTS, d_U, r_U);
  InterpTransposeTensor2d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
  WriteElementStrided2d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
}

//------------------------------------------------------------------------------
// Grad kernel by dim
//------------------------------------------------------------------------------
kernel void Grad(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_grad_1d, 
  global const CeedScalar * restrict d_U,
  global CeedScalar * restrict d_V) {

  local CeedScalar s_B[BASIS_Q_1D * BASIS_P_1D]; // Todo, don't allocate s_B for dimension 1
  local CeedScalar s_G[BASIS_Q_1D * BASIS_P_1D];
  
  private CeedScalar r_U[BASIS_NUM_COMP];
  private CeedScalar r_V[BASIS_NUM_COMP * 2];
  
  local CeedScalar scratch[BASIS_GRAD_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * T_1D * T_1D;

  loadMatrix(BASIS_Q_1D * BASIS_P_1D, d_interp_1d, s_B);
  loadMatrix(BASIS_Q_1D * BASIS_P_1D, d_grad_1d, s_G);

  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  ReadElementStrided2d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
  GradTensor2d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
  WriteElementStrided2d(BASIS_NUM_COMP * 2, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
}

kernel void GradTranspose(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_grad_1d, 
  global const CeedScalar * restrict d_U,
  global CeedScalar * restrict d_V) {

  local CeedScalar s_B[BASIS_Q_1D * BASIS_P_1D]; // Todo, don't allocate s_B for dimension 1
  local CeedScalar s_G[BASIS_Q_1D * BASIS_P_1D];
  
  private CeedScalar r_U[BASIS_NUM_COMP * 2];
  private CeedScalar r_V[BASIS_NUM_COMP];
  
  local CeedScalar scratch[BASIS_GRAD_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * T_1D * T_1D;

  loadMatrix(BASIS_Q_1D * BASIS_P_1D, d_interp_1d, s_B);
  loadMatrix(BASIS_Q_1D * BASIS_P_1D, d_grad_1d, s_G);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  ReadElementStrided2d(BASIS_NUM_COMP * 2, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
  GradTransposeTensor2d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
  WriteElementStrided2d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
}

//------------------------------------------------------------------------------
// Weight kernels by dim
//------------------------------------------------------------------------------
kernel void Weight(const CeedInt num_elem, global const CeedScalar * restrict q_weight_1d, global CeedScalar * restrict d_W) {

  private CeedScalar r_W[1];
  // void prefetch(q_weight_1d,BASIS_Q_1D);
  WeightTensor2d(BASIS_Q_1D, q_weight_1d, r_W);
  WriteElementStrided2d(1, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_W, d_W);
}

//------------------------------------------------------------------------------

#endif
