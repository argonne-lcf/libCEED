// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL backend macro and type definitions for JiT source
#include <ceed/types.h>

// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

// TODO: Handle FP32 case
// typedef atomic_double CeedAtomicScalar;
typedef CeedScalar CeedAtomicScalar;
    
//------------------------------------------------------------------------------
// Load matrices for basis actions
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
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D>
inline void readDofsOffset1d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt *__restrict__ indices,
                             const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x;
    const CeedInt ind  = indices[node + data.item_id_z * P_1D];
    for (CeedInt comp = 0; comp < NUM_COMP; ++comp) {
      r_u[comp] = d_u[ind + COMP_STRIDE * comp];
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline void readDofsStrided1d(SharedData_Sycl &data, const CeedInt num_elem, const CeedScalar *__restrict__ d_u,
                              CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x;
    const CeedInt ind  = node * STRIDES_NODE + data.item_id_z * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * STRIDES_COMP];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D>
inline void writeDofsOffset1d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt *__restrict__ indices,
                              const CeedScalar *__restrict__ r_v, CeedAtomicScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x;
    const CeedInt ind  = indices[node + data.item_id_z * P_1D];
    for (CeedInt comp = 0; comp < NUM_COMP; ++comp) {
      // atomic_fetch_add_explicit(&d_v[ind + COMP_STRIDE * comp], r_v[comp], memory_order_relaxed, memory_scope_device);
      // SYCL atomic_ref
      // auto v = sycl::atomic_ref<CeedAtomicScalar, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(d_v[ind + COMP_STRIDE * comp]);
      // v.fetch_add(r_v[comp], memory_order_relaxed, memory_scope::device);

      sycl::atomic_ref<CeedAtomicScalar, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(d_v[ind + COMP_STRIDE * comp]) += r_v[comp];
    }   
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline void writeDofsStrided1d(SharedData_Sycl &data, const CeedInt num_elem, const CeedScalar *__restrict__ r_v,
                               CeedScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x;
    const CeedInt ind  = node * STRIDES_NODE + data.item_id_z * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * STRIDES_COMP] = r_v[comp]; //CHECK THIS
    }
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D>
inline void readDofsOffset2d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt *__restrict__ indices,
                             const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x + data.item_id_y * P_1D;
    const CeedInt ind  = indices[node + data.item_id_z * P_1D * P_1D];

    for (CeedInt comp = 0; comp < NUM_COMP; ++comp) r_u[comp] = d_u[ind + COMP_STRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline void readDofsStrided2d(SharedData_Sycl &data, const CeedInt num_elem, const CeedScalar *__restrict__ d_u,
                              CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x + data.item_id_y * P_1D;
    const CeedInt ind  = node * STRIDES_NODE + data.item_id_z * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; ++comp) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D>
inline void writeDofsOffset2d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt *__restrict__ indices, 
                              const CeedScalar *__restrict__ r_v, CeedAtomicScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x + data.item_id_y * P_1D;
    const CeedInt ind  = indices[node + data.item_id_z * P_1D * P_1D];
    for (CeedInt comp = 0; comp < NUM_COMP; ++comp)
      // atomic_fetch_add_explicit(&d_v[ind + strides_comp * comp], r_v[comp], memory_order_relaxed, memory_scope_device);
      sycl::atomic_ref<CeedAtomicScalar, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(d_v[ind + COMP_STRIDE * comp]) += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline void writeDofsStrided2d(SharedData_Sycl &data, const CeedInt num_elem, const CeedScalar *__restrict__ r_v,
                               CeedScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x + data.item_id_y * P_1D;
    const CeedInt ind  = node * STRIDES_NODE + data.item_id_z * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; ++comp) d_v[ind + comp * STRIDES_COMP] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D>
inline void readDofsOffset3d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt *__restrict__ indices,
                             const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = data.item_id_x + P_1D * (data.item_id_y + P_1D * z);
      const CeedInt ind  = indices[node + data.item_id_z * P_1D * P_1D * P_1D];

      for (CeedInt comp = 0; comp < NUM_COMP; ++comp) r_u[z + comp * P_1D] = d_u[ind + COMP_STRIDE * comp];
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline void readDofsStrided3d(SharedData_Sycl &data, const CeedInt num_elem, const CeedScalar *__restrict__ d_u,
                              CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = data.item_id_x + P_1D * (data.item_id_y + P_1D * z);
      const CeedInt ind  = node * STRIDES_NODE + data.item_id_z * STRIDES_ELEM;

      for (CeedInt comp = 0; comp < NUM_COMP; ++comp) r_u[z + comp * P_1D] = d_u[ind + comp * STRIDES_COMP];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int Q_1D>
inline void readSliceQuadsOffset3d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt q,
                                   const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < Q_1D && data.item_id_y < Q_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x + Q_1D * (data.item_id_y + Q_1D * q);
    const CeedInt ind  = indices[node + data.item_id_z * Q_1D * Q_1D * Q_1D];

    for (CeedInt comp = 0; comp < NUM_COMP; ++comp) r_u[comp] = d_u[ind + COMP_STRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline void readSliceQuadsStrided3d(SharedData_Sycl &data, const CeedInt num_elem, const CeedInt q, const CeedScalar *__restrict__ d_u,
                                    CeedScalar *__restrict__ r_u) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < Q_1D && data.item_id_y < Q_1D && data.item_id_z < num_elem) {
    const CeedInt node = data.item_id_x + Q_1D * (data.item_id_y + Q_1D * q);
    const CeedInt ind  = node * STRIDES_NODE + data.item_id_z * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NUM_COMP; ++comp) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D>
inline void writeDofsOffset3d(SharedData_Sycl &data, const CeedInt num_elem,
                              const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ r_v, CeedAtomicScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = data.item_id_x + data.item_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = indices[node + data.item_id_z * P_1D * P_1D * P_1D];

      for (CeedInt comp = 0; comp < NUM_COMP; ++comp)
        // atomic_fetch_add_explicit(&d_v[ind + strides_comp * comp], r_v[z + comp * P_1D], memory_order_relaxed, memory_scope_device);
        sycl::atomic_ref<CeedAtomicScalar, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(d_v[ind + COMP_STRIDE * comp]) += r_v[z + comp * P_1D];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline void writeDofsStrided3d(SharedData_Sycl &data, const CeedInt num_elem, const CeedScalar *__restrict__ r_v,
                               CeedScalar *__restrict__ d_v) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);
  // const CeedInt elem      = get_global_id(2);

  if (data.item_id_x < P_1D && data.item_id_y < P_1D && data.item_id_z < num_elem) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = data.item_id_x + P_1D * (data.item_id_y + P_1D * z);
      const CeedInt ind  = node * STRIDES_NODE + data.item_id_z * STRIDES_ELEM;

      for (CeedInt comp = 0; comp < NUM_COMP; ++comp) d_v[ind + comp * STRIDES_COMP] += r_v[z + comp * P_1D];
    }
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives computation
//------------------------------------------------------------------------------
template<int NUM_COMP, int Q_1D>
inline void gradCollo3d(SharedData_Sycl &data, sycl::nd_item<3> &work_item, const CeedInt q, const CeedScalar *__restrict__ r_U,
                        const CeedScalar *s_G, CeedScalar *__restrict__ r_V) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  for (CeedInt comp = 0; comp < NUM_COMP; ++comp) {
    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      data.scratch[data.item_id_x + data.item_id_y * T_1D] = r_U[q + comp * Q_1D];
    }
    work_item.barrier(sycl::access::fence_space::local_space);

    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      // X derivative
      r_V[comp + 0 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[comp + 0 * NUM_COMP] += s_G[i + data.item_id_x * Q_1D] * data.scratch[i + data.item_id_y * T_1D];  // Contract x direction (X derivative)

      // Y derivative
      r_V[comp + 1 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[comp + 1 * NUM_COMP] += s_G[i + data.item_id_y * Q_1D] * data.scratch[data.item_id_x + i * T_1D];  // Contract y direction (Y derivative)

      // Z derivative
      r_V[comp + 2 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1D; ++i) r_V[comp + 2 * NUM_COMP] += s_G[i + q * Q_1D] * r_U[i + comp * Q_1D];  // Contract z direction (Z derivative)
    }

    work_item.barrier(sycl::access::fence_space::local_space);
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives transpose
//------------------------------------------------------------------------------
template<int NUM_COMP, int Q_1D>
inline void gradColloTranspose3d(SharedData_Sycl &data, sycl::nd_item<3> &work_item, const CeedInt q, const CeedScalar *__restrict__ r_U,
                                 const CeedScalar *__restrict__ s_G, CeedScalar *__restrict__ r_V) {
  // const CeedInt item_id_x = get_local_id(0);
  // const CeedInt item_id_y = get_local_id(1);

  for (CeedInt comp = 0; comp < NUM_COMP; ++comp) {
    // X derivative
    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      data.scratch[data.item_id_x + data.item_id_y * T_1D] = r_U[comp + 0 * NUM_COMP];
    }
    work_item.barrier(sycl::access::fence_space::local_space);

    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[q + comp * Q_1D] += s_G[data.item_id_x + i * Q_1D] * data.scratch[i + data.item_id_y * T_1D];  // Contract x direction (X derivative)
    }
    work_item.barrier(sycl::access::fence_space::local_space);

    // Y derivative
    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      data.scratch[data.item_id_x + data.item_id_y * T_1D] = r_U[comp + 1 * NUM_COMP];
    }
    work_item.barrier(sycl::access::fence_space::local_space);

    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[q + comp * Q_1D] += s_G[data.item_id_y + i * Q_1D] * data.scratch[data.item_id_x + i * T_1D];  // Contract y direction (Y derivative)
    }
    work_item.barrier(sycl::access::fence_space::local_space);

    // Z derivative
    if (data.item_id_x < Q_1D && data.item_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[i + comp * Q_1D] += s_G[i + q * Q_1D] * r_U[comp + 2 * NUM_COMP];  // PARTIAL contract z direction (Z derivative)
    }
  }
}
