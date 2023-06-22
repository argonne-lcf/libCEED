// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory tensor product basis
#ifndef _ceed_sycl_shared_basis_tensor_1d_h
#define _ceed_sycl_shared_basis_tensor_1d_h

#include <ceed.h>

#include "sycl-shared-basis-read-write-templates-1d.h"
#include "sycl-shared-basis-tensor-templates-1d.h"

//
// BASIS_NUM_NODES = CeedIntPow(BASIS_P_1D)
// BASIS_NUM_QPTS = CeedIntPow(BASIS_Q_1D)


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
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * T_1D;

  loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  ReadElementStrided1d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
  Interp1d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
  WriteElementStrided1d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
}

kernel void InterpTranspose(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_U,
  global CeedScalar * restrict d_V) {
  
  local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  private CeedScalar r_U[BASIS_NUM_COMP];
  private CeedScalar r_V[BASIS_NUM_COMP];
  
  local CeedScalar scratch[BASIS_INTERP_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * T_1D;

  loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  ReadElementStrided1d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
  InterpTranspose1d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
  WriteElementStrided1d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
}

//------------------------------------------------------------------------------
// Grad kernel by dim
//------------------------------------------------------------------------------
kernel void Grad(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_grad_1d, 
  global const CeedScalar * restrict d_U,
  global CeedScalar * restrict d_V) {

  local CeedScalar s_G[BASIS_Q_1D * BASIS_P_1D];
  
  private CeedScalar r_U[BASIS_NUM_COMP];
  private CeedScalar r_V[BASIS_NUM_COMP];
  
  local CeedScalar scratch[BASIS_GRAD_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * T_1D;

  loadMatrix(BASIS_Q_1D * BASIS_P_1D, d_grad_1d, s_G);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  ReadElementStrided1d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
  Grad1d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_G, r_V, elem_scratch);
  WriteElementStrided1d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
}

kernel void GradTranspose(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_grad_1d, 
  global const CeedScalar * restrict d_U,
  global CeedScalar * restrict d_V) {

  local CeedScalar s_G[BASIS_Q_1D * BASIS_P_1D];
  
  private CeedScalar r_U[BASIS_NUM_COMP];
  private CeedScalar r_V[BASIS_NUM_COMP];
  
  local CeedScalar scratch[BASIS_GRAD_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * T_1D;

  loadMatrix(BASIS_Q_1D * BASIS_P_1D, d_grad_1d, s_G);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  ReadElementStrided1d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
  GradTranspose1d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_G, r_V, elem_scratch);
  WriteElementStrided1d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
}

//------------------------------------------------------------------------------
// Weight kernels by dim
//------------------------------------------------------------------------------
kernel void Weight(const CeedInt num_elem, global const CeedScalar * restrict q_weight_1d, global CeedScalar * restrict d_W) {

  private CeedScalar r_W[1];

  // void prefetch(q_weight_1d,BASIS_Q_1D);
  Weight1d(BASIS_Q_1D, q_weight_1d, r_W);
  WriteElementStrided1d(1, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_W, d_W);
}

//------------------------------------------------------------------------------

#endif
