// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory tensor product basis templates
#include <ceed/types.h>

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 1D tensor contraction x
//------------------------------------------------------------------------------
// Umesh: NUM_COMP not necessary in Contract functions
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractX1d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                        CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);

  data.scratch[data.item_id_x] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (data.item_id_x < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.item_id_x * P_1D] * data.scratch[i];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractTransposeX1d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                                 CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);

  data.scratch[data.item_id_x] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (data.item_id_x < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.item_id_x + i * P_1D] * data.scratch[i];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void Interp1d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                     const CeedScalar *restrict s_B, CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d<NUM_COMP, P_1D, Q_1D> (data, r_U + comp, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void InterpTranspose1d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                              const CeedScalar *restrict s_B, CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void Grad1d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                   const CeedScalar *restrict s_G, CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, s_G, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void GradTranspose1d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                            const CeedScalar *restrict s_G, CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, s_G, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D quadrature weights
//------------------------------------------------------------------------------
template <int Q_1D>
inline void Weight1d(SharedData_Sycl &data, const CeedScalar *restrict q_weight_1d, CeedScalar *restrict w) {
  // const CeedInt item_id_x = get_local_id(0);
  *w                      = (data.item_id_x < Q_1D) ? q_weight_1d[data.item_id_x] : 0.0;
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractX2d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                        CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  data.scratch[data.item_id_x + data.item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (data.item_id_x < Q_1D && data.item_id_y < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.item_id_x * P_1D] * data.scratch[i + data.item_id_y * T_1D];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractY2d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                        CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  data.scratch[data.item_id_x + data.item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.item_id_y * P_1D] * data.scratch[data.item_id_x + i * T_1D];  // Contract y direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractTransposeY2d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                                 CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  data.scratch[data.item_id_x + data.item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (data.item_id_x < Q_1D && data.item_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.item_id_y + i * P_1D] * data.scratch[data.item_id_x + i * T_1D];  // Contract y direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractTransposeX2d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                                 CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  data.scratch[data.item_id_x + data.item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (data.item_id_x < P_1D && data.item_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.item_id_x + i * P_1D] * data.scratch[i + data.item_id_y * T_1D];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract and add x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractTransposeAddX2d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                                    CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  data.scratch[data.item_id_x + data.item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.item_id_x + i * P_1D] * data.scratch[i + data.item_id_y * T_1D];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void InterpTensor2d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                           const CeedScalar *restrict s_B, CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t[1];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, s_B, r_t, scratch);
    ContractY2d<NUM_COMP, P_1D, Q_1D>(data, r_t, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 2D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void InterpTransposeTensor2d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                                    const CeedScalar *restrict s_B, CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t[1];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, s_B, r_t, scratch);
    ContractTransposeX2d<NUM_COMP, P_1D, Q_1D>(data, r_t, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void GradTensor2d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                         const CeedScalar *restrict s_B, const CeedScalar *restrict s_G, CeedScalar *restrict r_V,
                         local CeedScalar *restrict scratch) {
  CeedScalar r_t[1];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, s_G, r_t, scratch);
    ContractY2d<NUM_COMP, P_1D, Q_1D>(data, r_t, s_B, r_V + comp + 0 * NUM_COMP, scratch);
    ContractX2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, s_B, r_t, scratch);
    ContractY2d<NUM_COMP, P_1D, Q_1D>(data, r_t, s_G, r_V + comp + 1 * NUM_COMP, scratch);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void GradTransposeTensor2d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                                  const CeedScalar *restrict s_B, const CeedScalar *restrict s_G, CeedScalar *restrict r_V,
                                  local CeedScalar *restrict scratch) {
  CeedScalar r_t[1];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp + 0 * NUM_COMP, s_B, r_t, scratch);
    ContractTransposeX2d<NUM_COMP, P_1D, Q_1D>(data, r_t, s_G, r_V + comp, scratch);
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp + 1 * NUM_COMP, s_G, r_t, scratch);
    ContractTransposeAddX2d<NUM_COMP, P_1D, Q_1D>(data, r_t, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
template <int Q_1D>
inline void WeightTensor2d(SharedData_Sycl &data, const CeedScalar *restrict q_weight_1d, CeedScalar *restrict w) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  *w = (data.item_id_x < Q_1D && data.item_id_y < Q_1D) ? q_weight_1d[data.item_id_x] * q_weight_1d[data.item_id_y] : 0.0;
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractX3d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                        CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < P_1D; i++) {
    r_B[i] = B[i + data.item_id_x * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    data.scratch[data.item_id_x + data.item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    V[k] = 0.0;
    if (data.item_id_x < Q_1D && data.item_id_y < P_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * data.scratch[i + data.item_id_y * T_1D];  // Contract x direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractY3d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                        CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < P_1D; i++) {
    r_B[i] = B[i + data.item_id_y * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    data.scratch[data.item_id_x + data.item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    V[k] = 0.0;
    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * data.scratch[data.item_id_x + i * T_1D];  // Contract y direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractZ3d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                        CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  for (CeedInt k = 0; k < Q_1D; k++) {
    V[k] = 0.0;
    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += B[i + k * P_1D] * U[i];  // Contract z direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractTransposeZ3d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                                 CeedScalar *restrict V, CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  for (CeedInt k = 0; k < P_1D; k++) {
    V[k] = 0.0;
    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += B[k + i * P_1D] * U[i];  // Contract z direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractTransposeY3d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                                 CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.item_id_y + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    data.scratch[data.item_id_x + data.item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    V[k] = 0.0;
    if (data.item_id_x < Q_1D && data.item_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.scratch[data.item_id_x + i * T_1D];  // Contract y direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractTransposeAddY3d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                                    CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.item_id_y + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    data.scratch[data.item_id_x + data.item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    if (data.item_id_x < Q_1D && data.item_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.scratch[data.item_id_x + i * T_1D];  // Contract y direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractTransposeX3d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                                 CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.item_id_x + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    data.scratch[data.item_id_x + data.item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    V[k] = 0.0;
    if (data.item_id_x < P_1D && data.item_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.scratch[i + data.item_id_y * T_1D];  // Contract x direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract add x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void ContractTransposeAddX3d(SharedData_Sycl &data, const CeedScalar *restrict U, const CeedScalar *restrict B,
                                    CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.item_id_x + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    data.scratch[data.item_id_x + data.item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (data.item_id_x < P_1D && data.item_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.scratch[i + data.item_id_y * T_1D];  // Contract x direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void InterpTensor3d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                           const CeedScalar *restrict s_B, CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp * P_1D, s_B, r_t1, scratch);
    ContractY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_B, r_t2, scratch);
    ContractZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_B, r_V + comp * Q_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void InterpTransposeTensor3d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                                    const CeedScalar *restrict s_B, CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp * Q_1D, s_B, r_t1, scratch);
    ContractTransposeY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_B, r_t2, scratch);
    ContractTransposeX3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_B, r_V + comp * P_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void GradTensor3d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                         const CeedScalar *restrict s_B, const CeedScalar *restrict s_G, CeedScalar *restrict r_V,
                         local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp * P_1D, s_G, r_t1, scratch);
    ContractY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_B, r_t2, scratch);
    ContractZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_B, r_V + comp * Q_1D + 0 * NUM_COMP * Q_1D, scratch);
    ContractX3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp * P_1D, s_B, r_t1, scratch);
    ContractY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_G, r_t2, scratch);
    ContractZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_B, r_V + comp * Q_1D + 1 * NUM_COMP * Q_1D, scratch);
    ContractX3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp * P_1D, s_B, r_t1, scratch);
    ContractY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_B, r_t2, scratch);
    ContractZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_G, r_V + comp * Q_1D + 2 * NUM_COMP * Q_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void GradTransposeTensor3d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                                  const CeedScalar *restrict s_B, const CeedScalar *restrict s_G, CeedScalar *restrict r_V,
                                  local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp * Q_1D + 0 * NUM_COMP * Q_1D, s_B, r_t1, scratch);
    ContractTransposeY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_B, r_t2, scratch);
    ContractTransposeX3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_G, r_V + comp * P_1D, scratch);
    ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp * Q_1D + 1 * NUM_COMP * Q_1D, s_B, r_t1, scratch);
    ContractTransposeY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_G, r_t2, scratch);
    ContractTransposeAddX3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_B, r_V + comp * P_1D, scratch);
    ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp * Q_1D + 2 * NUM_COMP * Q_1D, s_G, r_t1, scratch);
    ContractTransposeY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_B, r_t2, scratch);
    ContractTransposeAddX3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_B, r_V + comp * P_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void GradTensorCollocated3d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                                   const CeedScalar *restrict s_B, const CeedScalar *restrict s_G, CeedScalar *restrict r_V,
                                   local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp * P_1D, s_B, r_t1, scratch);
    ContractY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_B, r_t2, scratch);
    ContractZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_B, r_t1, scratch);
    ContractX3d<NUM_COMP, Q_1D, Q_1D>(data, r_t1, s_G, r_V + comp * Q_1D + 0 * NUM_COMP * Q_1D, scratch);
    ContractY3d<NUM_COMP, Q_1D, Q_1D>(data, r_t1, s_G, r_V + comp * Q_1D + 1 * NUM_COMP * Q_1D, scratch);
    ContractZ3d<NUM_COMP, Q_1D, Q_1D>(data, r_t1, s_G, r_V + comp * Q_1D + 2 * NUM_COMP * Q_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline void GradTransposeTensorCollocated3d(SharedData_Sycl &data, const CeedScalar *restrict r_U,
                                            const CeedScalar *restrict s_B, const CeedScalar *restrict s_G,
                                            CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3d<NUM_COMP, Q_1D, Q_1D>(data, r_U + comp * Q_1D + 2 * NUM_COMP * Q_1D, s_G, r_t2, scratch);
    ContractTransposeAddY3d<NUM_COMP, Q_1D, Q_1D>(data, r_U + comp * Q_1D + 1 * NUM_COMP * Q_1D, s_G, r_t2, scratch);
    ContractTransposeAddX3d<NUM_COMP, Q_1D, Q_1D>(data, r_U + comp * Q_1D + 0 * NUM_COMP * Q_1D, s_G, r_t2, scratch);
    ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_B, r_t1, scratch);
    ContractTransposeY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, s_B, r_t2, scratch);
    ContractTransposeX3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, s_B, r_V + comp * P_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
template <int Q_1D>
inline void WeightTensor3d(SharedData_Sycl &data, const CeedScalar *restrict q_weight_1d, CeedScalar *restrict w) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  const bool       quad = (data.item_id_x < Q_1D && data.item_id_y < Q_1D);
  const CeedScalar pw   = quad ? q_weight_1d[data.item_id_x] * q_weight_1d[data.item_id_y] : 0.0;

  for (CeedInt q = 0; q < Q_1D; ++q) {
    w[q] = quad ? pw * q_weight_1d[q] : 0.0;
    // Umesh: This can be replaced without the conditional operator
  }
}
