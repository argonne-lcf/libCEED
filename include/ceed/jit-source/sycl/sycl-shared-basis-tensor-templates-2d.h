// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory tensor product basis templates
#ifndef _ceed_sycl_shared_basis_tensor_templates_2d_h
#define _ceed_sycl_shared_basis_tensor_templates_2d_h

#include <ceed.h>

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
inline void ContractX2d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + item_id_x * P_1D] * scratch[i + item_id_y * T_1D];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D tensor contract y
//------------------------------------------------------------------------------
inline void ContractY2d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + item_id_y * P_1D] * scratch[item_id_x + i * T_1D];  // Contract y direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract y
//------------------------------------------------------------------------------
inline void ContractTransposeY2d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_y + i * P_1D] * scratch[item_id_x + i * T_1D];  // Contract y direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract x
//------------------------------------------------------------------------------
inline void ContractTransposeX2d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {
  
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  
  *V = 0.0;
  if (item_id_x < P_1D && item_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_x + i * P_1D] * scratch[i + item_id_y * T_1D];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract and add x
//------------------------------------------------------------------------------
inline void ContractTransposeAddX2d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  if (item_id_x < P_1D && item_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_x + i * P_1D] * scratch[i + item_id_y * T_1D];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
inline void InterpTensor2d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t[1];

  ContractX2d(P_1D, Q_1D, r_U, s_B, r_t, scratch);
  ContractY2d(P_1D, Q_1D, r_t, s_B, r_V, scratch);
}

//------------------------------------------------------------------------------
// 2D interpolate transpose
//------------------------------------------------------------------------------
inline void InterpTransposeTensor2d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t[1];

  ContractTransposeY2d(P_1D, Q_1D, r_U, s_B, r_t, scratch);
  ContractTransposeX2d(P_1D, Q_1D, r_t, s_B, r_V, scratch);
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
inline void GradTensor2d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar *restrict r_U, 
  local const CeedScalar * restrict s_B, 
  local const CeedScalar * restrict s_G,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t[1];

  ContractX2d(P_1D, Q_1D, r_U, s_G, r_t, scratch);
  ContractY2d(P_1D, Q_1D, r_t, s_B, r_V, scratch);

  ContractX2d(P_1D, Q_1D, r_U, s_B, r_t, scratch);
  ContractY2d(P_1D, Q_1D, r_t, s_G, r_V + 1, scratch);
}

//------------------------------------------------------------------------------
// 2D derivatives transpose
//------------------------------------------------------------------------------
inline void GradTransposeTensor2d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B, 
  local const CeedScalar * restrict s_G,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t[1];

  ContractTransposeY2d(P_1D, Q_1D, r_U, s_B, r_t, scratch);
  ContractTransposeX2d(P_1D, Q_1D, r_t, s_G, r_V, scratch);

  ContractTransposeY2d(P_1D, Q_1D, r_U + 1, s_G, r_t, scratch);
  ContractTransposeAddX2d(P_1D, Q_1D, r_t, s_B, r_V, scratch);
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
inline void WeightTensor2d(const CeedInt Q_1D, const CeedScalar * restrict q_weight_1d, CeedScalar * restrict w) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  *w = (item_id_x < Q_1D && item_id_y < Q_1D) ? q_weight_1d[item_id_x] * q_weight_1d[item_id_y] : 0.0;
}

//------------------------------------------------------------------------------

#endif
