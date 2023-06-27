// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory tensor product basis
#ifndef _ceed_sycl_shared_basis_tensor_3d_h
#define _ceed_sycl_shared_basis_tensor_3d_h

#include <ceed.h>

#include "sycl-shared-basis-read-write-templates-3d.h"
#include "sycl-shared-basis-tensor-templates-3d.h"

//
// BASIS_NUM_NODES = CeedIntPow(BASIS_P_1D,3)
// BASIS_NUM_QPTS = CeedIntPow(BASIS_Q_1D,3)


//------------------------------------------------------------------------------
// Interp kernel
//------------------------------------------------------------------------------
kernel void Interp(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_U, 
  global CeedScalar * restrict d_V) {

  local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  private CeedScalar r_U[1];
  private CeedScalar r_V[1];
  
  local CeedScalar scratch[BASIS_INTERP_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * (T_1D * T_1D * T_1D);

  loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  
  for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
    ReadElementStrided3d(comp, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U,r_U);
    InterpTensor3d(BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
    WriteElementStrided3d(comp, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V,d_V);
  }
}

kernel void InterpTranspose(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_U,
  global CeedScalar * restrict d_V) {

  local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  private CeedScalar r_U[1];
  private CeedScalar r_V[1];
  
  local CeedScalar scratch[BASIS_INTERP_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * (T_1D * T_1D * T_1D);

  loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
    ReadElementStrided3d(comp, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
    InterpTransposeTensor3d(BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
    WriteElementStrided3d(comp, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
  }
}

//------------------------------------------------------------------------------
// Grad kernel
//------------------------------------------------------------------------------
kernel void Grad(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_grad_1d, 
  global const CeedScalar * restrict d_U,
  global CeedScalar * restrict d_V) {

  local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  local CeedScalar s_G[BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D)];
  
  private CeedScalar r_U[1];
  private CeedScalar r_V[3];
  
  local CeedScalar scratch[BASIS_GRAD_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * (T_1D * T_1D * T_1D);

  loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
  loadMatrix(BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D), d_grad_1d, s_G);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
    ReadElementStrided3d(comp, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
    
    if (BASIS_HAS_COLLOCATED_GRAD) GradTensorCollocated3d(BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
    else GradTensor3d(BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
    
    for (CeedInt dim = 0; dim < 3; ++dim) {
      WriteElementStrided3d(comp + BASIS_NUM_COMP * dim, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V + dim, d_V);
    }
  }
}

kernel void GradTranspose(const CeedInt num_elem, 
  global const CeedScalar * restrict d_interp_1d, 
  global const CeedScalar * restrict d_grad_1d, 
  global const CeedScalar * restrict d_U,
  global CeedScalar * restrict d_V) {

  local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D]; // Todo, don't allocate s_B for dimension 1
  local CeedScalar s_G[BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D)];
  
  private CeedScalar r_U[3];
  private CeedScalar r_V[1];
  
  local CeedScalar scratch[BASIS_GRAD_SCRATCH_SIZE];
  local CeedScalar * elem_scratch  = scratch + get_local_id(2) * (T_1D * T_1D * T_1D);

  loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
  loadMatrix(BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D), d_grad_1d, s_G);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
    for (CeedInt dim = 0; dim < 3; ++dim) {
      ReadElementStrided3d(comp + BASIS_NUM_COMP * dim, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U + dim);
    }

    if (BASIS_HAS_COLLOCATED_GRAD) GradTransposeTensorCollocated3d(BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
    else GradTransposeTensor3d(BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
    
    WriteElementStrided3d(comp, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
  }
}

//------------------------------------------------------------------------------
// Weight kernel
//------------------------------------------------------------------------------
kernel void Weight(const CeedInt num_elem, global const CeedScalar * restrict q_weight_1d, global CeedScalar * restrict d_W) {

  private CeedScalar r_W[1];
  // void prefetch(q_weight_1d,BASIS_Q_1D);
  WeightTensor3d(BASIS_Q_1D, q_weight_1d, r_W);
  WriteElementStrided3d(0, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_W, d_W);
}

//------------------------------------------------------------------------------

#endif
