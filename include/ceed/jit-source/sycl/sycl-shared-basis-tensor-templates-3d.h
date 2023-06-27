// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory tensor product basis templates
#ifndef _ceed_sycl_shared_basis_tensor_templates_3d_h
#define _ceed_sycl_shared_basis_tensor_templates_3d_h

#include <ceed.h>

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
inline void ContractX3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1) % T_1D;
  const CeedInt item_id_z = get_local_id(1) / T_1D;

  scratch[item_id_x + T_1D * (item_id_y + T_1D * item_id_z)] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  (*V) = 0.0;
  if (item_id_x < Q_1D && item_id_y < P_1D && item_id_z < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      (*V) += B[i + item_id_x * P_1D] * scratch[i + T_1D * (item_id_y + T_1D * item_id_z)];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
inline void ContractY3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1) % T_1D;
  const CeedInt item_id_z = get_local_id(1) / T_1D;

  scratch[item_id_x + T_1D * (item_id_y + T_1D * item_id_z)] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < Q_1D && item_id_z < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V +=  B[i + item_id_y * P_1D] * scratch[item_id_x + T_1D * (i + T_1D * item_id_z)];  // Contract y direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
inline void ContractZ3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {
  
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1) % T_1D;
  const CeedInt item_id_z = get_local_id(1) / T_1D;

  scratch[item_id_x + T_1D * (item_id_y + T_1D * item_id_z)] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < Q_1D && item_id_z < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + item_id_z * P_1D] * scratch[item_id_x + T_1D * (item_id_y + T_1D * i)];  // Contract z direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
inline void ContractTransposeZ3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1) % T_1D;
  const CeedInt item_id_z = get_local_id(1) / T_1D;

  scratch[item_id_x + T_1D * (item_id_y + T_1D * item_id_z)] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < Q_1D && item_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_z + i * P_1D] * scratch[item_id_x + T_1D * (item_id_y + T_1D * i)];  // Contract z direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
inline void ContractTransposeY3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1) % T_1D;
  const CeedInt item_id_z = get_local_id(1) / T_1D;

  scratch[item_id_x + T_1D * (item_id_y + T_1D * item_id_z)] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < P_1D && item_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_y + i * P_1D] * scratch[item_id_x + T_1D * (i + T_1D * item_id_z)];  // Contract y direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
inline void ContractTransposeAddY3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1) % T_1D;
  const CeedInt item_id_z = get_local_id(1) / T_1D;

  scratch[item_id_x + T_1D * (item_id_y + T_1D * item_id_z)] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  if (item_id_x < Q_1D && item_id_y < P_1D && item_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_y + i * P_1D] * scratch[item_id_x + T_1D * (i + T_1D * item_id_z)];  // Contract y direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
inline void ContractTransposeX3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1) % T_1D;
  const CeedInt item_id_z = get_local_id(1) / T_1D;

  scratch[item_id_x + T_1D * (item_id_y + T_1D * item_id_z)] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  
  *V = 0.0;
  if (item_id_x < P_1D && item_id_y < P_1D && item_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_x + i * P_1D] * scratch[i + T_1D * (item_id_y + T_1D * item_id_z)];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract add x
//------------------------------------------------------------------------------
inline void ContractTransposeAddX3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict U, 
  local const CeedScalar * restrict B, 
  private CeedScalar * restrict V,
  local CeedScalar * restrict scratch) {

  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1) % T_1D;
  const CeedInt item_id_z = get_local_id(1) / T_1D;

  scratch[item_id_x + T_1D * (item_id_y + T_1D * item_id_z)] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  
  if (item_id_x < P_1D && item_id_y < P_1D && item_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_x + i * P_1D] * scratch[i + T_1D * (item_id_y + T_1D * item_id_z)];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
inline void InterpTensor3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t1[1];
  CeedScalar r_t2[1];

  ContractX3d(P_1D, Q_1D, r_U, s_B, r_t1, scratch);
  ContractY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
  ContractZ3d(P_1D, Q_1D, r_t2, s_B, r_V, scratch);
}

//------------------------------------------------------------------------------
// 3D interpolate transpose
//------------------------------------------------------------------------------
inline void InterpTransposeTensor3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t1[1];
  CeedScalar r_t2[1];

  ContractTransposeZ3d(P_1D, Q_1D, r_U, s_B, r_t1, scratch);
  ContractTransposeY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
  ContractTransposeX3d(P_1D, Q_1D, r_t2, s_B, r_V, scratch);
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
inline void GradTensor3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B, 
  local const CeedScalar * restrict s_G,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t1[1];
  CeedScalar r_t2[1];

  ContractX3d(P_1D, Q_1D, r_U, s_G, r_t1, scratch);
  ContractY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
  ContractZ3d(P_1D, Q_1D, r_t2, s_B, r_V + 0, scratch);

  ContractX3d(P_1D, Q_1D, r_U, s_B, r_t1, scratch);
  ContractY3d(P_1D, Q_1D, r_t1, s_G, r_t2, scratch);
  ContractZ3d(P_1D, Q_1D, r_t2, s_B, r_V + 1, scratch);

  ContractX3d(P_1D, Q_1D, r_U, s_B, r_t1, scratch);
  ContractY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
  ContractZ3d(P_1D, Q_1D, r_t2, s_G, r_V + 2, scratch);
}

// //------------------------------------------------------------------------------
// // 3D derivatives transpose
// //------------------------------------------------------------------------------
inline void GradTransposeTensor3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B, 
  local const CeedScalar * restrict s_G,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t1[1];
  CeedScalar r_t2[1];

  ContractTransposeZ3d(P_1D, Q_1D, r_U + 0, s_B, r_t1, scratch);
  ContractTransposeY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
  ContractTransposeX3d(P_1D, Q_1D, r_t2, s_G, r_V, scratch);

  ContractTransposeZ3d(P_1D, Q_1D, r_U + 1, s_B, r_t1, scratch);
  ContractTransposeY3d(P_1D, Q_1D, r_t1, s_G, r_t2, scratch);
  ContractTransposeAddX3d(P_1D, Q_1D, r_t2, s_B, r_V, scratch);
  
  ContractTransposeZ3d(P_1D, Q_1D, r_U + 2, s_G, r_t1, scratch);
  ContractTransposeY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
  ContractTransposeAddX3d(P_1D, Q_1D, r_t2, s_B, r_V, scratch);
}

// //------------------------------------------------------------------------------
// // 3D derivatives at quadrature points
// //------------------------------------------------------------------------------
inline void GradTensorCollocated3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B, 
  local const CeedScalar * restrict s_G,
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t1[1];
  CeedScalar r_t2[1];
  
  ContractX3d(P_1D, Q_1D, r_U, s_B, r_t1, scratch);
  ContractY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
  ContractZ3d(P_1D, Q_1D, r_t2, s_B, r_t1, scratch);
  ContractX3d(Q_1D, Q_1D, r_t1, s_G, r_V, scratch);
  ContractY3d(Q_1D, Q_1D, r_t1, s_G, r_V + 1, scratch);
  ContractZ3d(Q_1D, Q_1D, r_t1, s_G, r_V + 2, scratch);
}

// //------------------------------------------------------------------------------
// // 3D derivatives transpose
// //------------------------------------------------------------------------------
inline void GradTransposeTensorCollocated3d(const CeedInt P_1D, const CeedInt Q_1D,
  private const CeedScalar * restrict r_U, 
  local const CeedScalar * restrict s_B,
  local const CeedScalar * restrict s_G, 
  private CeedScalar * restrict r_V,
  local CeedScalar * restrict scratch) {

  CeedScalar r_t1[1];
  CeedScalar r_t2[1];

  ContractTransposeZ3d(Q_1D, Q_1D, r_U + 2, s_G, r_t2, scratch);
  ContractTransposeAddY3d(Q_1D, Q_1D, r_U + 1, s_G, r_t2, scratch);
  ContractTransposeAddX3d(Q_1D, Q_1D, r_U, s_G, r_t2, scratch);
  ContractTransposeZ3d(P_1D, Q_1D, r_t2, s_B, r_t1, scratch);
  ContractTransposeY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
  ContractTransposeX3d(P_1D, Q_1D, r_t2, s_B, r_V, scratch);
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
// template <int Q_1D>
inline void WeightTensor3d(const CeedInt Q_1D, const CeedScalar *restrict q_weight_1d, CeedScalar * restrict w) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1) % T_1D;
  const CeedInt item_id_z = get_local_id(1) / T_1D;
  if (item_id_x < Q_1D && item_id_y < Q_1D && item_id_z < Q_1D)
    *w = q_weight_1d[item_id_x] * q_weight_1d[item_id_y] * q_weight_1d[item_id_z]; 
}

//------------------------------------------------------------------------------

#endif
