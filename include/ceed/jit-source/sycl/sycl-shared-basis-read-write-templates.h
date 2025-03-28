// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory basis read/write templates
#include <ceed/types.h>

//------------------------------------------------------------------------------
// Helper function: load matrices for basis actions
//------------------------------------------------------------------------------
template <int P, int Q>
inline void loadMatrix(SharedData_Sycl &data, const CeedScalar *__restrict__ d_B, CeedScalar *__restrict__ B) {
  // const CeedInt item_id    = get_local_linear_id();
  // const CeedInt group_size = get_local_size(0) * get_local_size(1) * get_local_size(2);

  for (CeedInt i = data.item_id; i < P * Q; i += data.group_size) B[i] = d_B[i];
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline void ReadElementStrided1d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt strides_node,
                                 const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ d_u,
                                 CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x;
    const CeedInt ind  = node * strides_node + data.item_id_z * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline void WriteElementStrided1d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt strides_node,
                                  const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ r_v,
                                  CeedScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x;
    const CeedInt ind  = node * strides_node + data.item_id_z * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] = r_v[comp];
    }
  }
}

template <int NUM_COMP, int P_1D>
inline void SumElementStrided1d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt strides_node,
                                const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ r_v, CeedScalar *__restrict__ d_v) {
  if (data.item_id_x < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x;
    const CeedInt ind  = node * strides_node + data.item_id_z * strides_elem;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] += r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline void ReadElementStrided2d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt strides_node,
                                 const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ d_u,
                                 CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x + data.item_id_y * P_1D;
    const CeedInt ind  = node * strides_node + data.item_id_z * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline void WriteElementStrided2d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt strides_node,
                                  const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ r_v,
                                  CeedScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x + data.item_id_y * P_1D;
    const CeedInt ind  = node * strides_node + data.item_id_z * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] = r_v[comp];
    }
  }
}

template <int NUM_COMP, int P_1D>
inline void SumElementStrided2d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt strides_node,
                                  const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ r_v,
                                  CeedScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x + data.item_id_y * P_1D;
    const CeedInt ind  = node * strides_node + data.item_id_z * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] += r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline void ReadElementStrided3d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt strides_node,
                                 const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ d_u,
                                 CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.item_id_x + data.item_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = node * strides_node + data.item_id_z * strides_elem;
      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        r_u[z + comp * P_1D] = d_u[ind + comp * strides_comp];
      }
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline void WriteElementStrided3d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt strides_node,
                                  const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ r_v,
                                  CeedScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.item_id_x + data.item_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = node * strides_node + data.item_id_z * strides_elem;
      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        d_v[ind + comp * strides_comp] = r_v[z + comp * P_1D];
      }
    }
  }
}

template <int NUM_COMP, int P_1D>
inline void SumElementStrided3d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt strides_node,
                                  const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ r_v,
                                  CeedScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.item_id_x + data.item_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = node * strides_node + data.item_id_z * strides_elem;
      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        d_v[ind + comp * strides_comp] += r_v[z + comp * P_1D];
      }
    }
  }
}
