// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory tensor product basis templates
#ifndef _ceed_sycl_shared_basis_tensor_templates_1d_h
#define _ceed_sycl_shared_basis_tensor_templates_1d_h

#include <ceed.h>

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 1D tensor contraction x
//------------------------------------------------------------------------------
inline void ContractX1d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);

  scratch[item_id_x] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  
  *V = 0.0;
  if (item_id_x < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + item_id_x * P_1D] * scratch[i];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction x
//------------------------------------------------------------------------------
inline void ContractTransposeX1d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  
  scratch[item_id_x] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  
  *V = 0.0;
  if (item_id_x < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_x + i * P_1D] * scratch[i];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
inline void Interp1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B, 
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d(P_1D, Q_1D, r_U + comp, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D interpolate transpose
//------------------------------------------------------------------------------
inline void InterpTranspose1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d(P_1D, Q_1D, r_U + comp, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
inline void Grad1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_G,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d(P_1D, Q_1D, r_U + comp, s_G, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D derivatives transpose
//------------------------------------------------------------------------------
inline void GradTranspose1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_G,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d(P_1D, Q_1D, r_U + comp, s_G, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D quadrature weights
//------------------------------------------------------------------------------
inline void Weight1d(const CeedInt Q_1D, const CeedScalar * restrict q_weight_1d, CeedScalar * restrict w) {
  const CeedInt item_id_x = get_local_id(0);
  *w = (item_id_x < Q_1D) ? q_weight_1d[item_id_x] : 0.0;
}

//------------------------------------------------------------------------------

#endif
