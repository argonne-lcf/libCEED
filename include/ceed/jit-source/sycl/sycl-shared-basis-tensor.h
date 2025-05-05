// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory tensor product basis
#include <ceed/types.h>

#include "sycl-shared-basis-read-write-templates.h"
#include "sycl-shared-basis-tensor-templates.h"

template <int dim, int P, int Q> class CeedSyclSharedBasis_Interp;
template <int dim, int P, int Q> class CeedSyclSharedBasis_InterpTranspose;
template <int dim, int P, int Q> class CeedSyclSharedBasis_InterpTransposeAdd;
template <int dim, int P, int Q> class CeedSyclSharedBasis_Grad;
template <int dim, int P, int Q> class CeedSyclSharedBasis_GradTranspose;
template <int dim, int P, int Q> class CeedSyclSharedBasis_GradTransposeAdd;
template <int dim, int Q> class CeedSyclSharedBasis_Weight;

//
// BASIS_NUM_NODES = CeedIntPow(BASIS_P_1D,DIM)
// BASIS_NUM_QPTS = CeedIntPow(BASIS_Q_1D,DIM)

//------------------------------------------------------------------------------
// Interp kernel by dim
//------------------------------------------------------------------------------
// kernel void Interp(const CeedInt num_elem, global const CeedScalar *restrict d_interp_1d, global const CeedScalar *restrict d_U,
//                    global CeedScalar *restrict d_V) {
//   local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
//  private
//   CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
//  private
//   CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

//   local CeedScalar  scratch[BASIS_INTERP_SCRATCH_SIZE];
//   local CeedScalar *elem_scratch = scratch + get_local_id(2) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

//   loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
//   work_group_barrier(CLK_LOCAL_MEM_FENCE);

//   if (BASIS_DIM == 1) {
//     ReadElementStrided1d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
//     Interp1d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
//     WriteElementStrided1d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);

//   } else if (BASIS_DIM == 2) {
//     ReadElementStrided2d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
//     InterpTensor2d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
//     WriteElementStrided2d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);

//   } else if (BASIS_DIM == 3) {
//     ReadElementStrided3d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
//     InterpTensor3d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
//     WriteElementStrided3d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
//   }
// }

extern "C" void Interp(sycl::queue &sycl_queue, sycl::nd_range<3> kernel_range, const CeedInt num_elem, const CeedScalar *__restrict__ d_interp_1d, const CeedScalar *__restrict__ d_U,
                       CeedScalar *__restrict__ d_V) {  
  
  std::vector<sycl::event> e;
  if (!sycl_queue.is_in_order()) e = {sycl_queue.ext_oneapi_submit_barrier()};

  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);

    sycl::local_accessor<CeedScalar> s_memS(BASIS_INTERP_SCRATCH_SIZE, cgh);
    sycl::local_accessor<CeedScalar> s_memB(BASIS_P_1D * BASIS_Q_1D, cgh);

    cgh.parallel_for<CeedSyclSharedBasis_Interp<BASIS_DIM,BASIS_P_1D,BASIS_Q_1D>>(kernel_range, [=](sycl::nd_item<3> item) {
      CeedScalar *scratch = s_memS.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_B     = s_memB.get_multi_ptr<sycl::access::decorated::yes>().get();

      SharedData_Sycl data;
      // data.work_item = item;
      data.item_id_x = item.get_local_id(2);
      data.item_id_y = item.get_local_id(1);
      data.item_id_z = item.get_global_id(0);
      data.item_id   = item.get_local_linear_id();
      data.group_size = item.get_local_range(0) * item.get_local_range(1) * item.get_local_range(2);
      data.scratch = scratch + item.get_local_id(0) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);
      
      CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
      CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
      
      // load interp_1d into shared memory
      loadMatrix<BASIS_P_1D, BASIS_Q_1D> (data, d_interp_1d, s_B);
      item.barrier(sycl::access::fence_space::local_space);
      // sycl::group_barrier(item.get_group());
      
      if (BASIS_DIM == 1) {
        ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
        Interp1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, r_V);
        WriteElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
      } else if (BASIS_DIM == 2) {
        ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
        InterpTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, r_V);
        WriteElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
      } else if (BASIS_DIM == 3) {
        ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
        InterpTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, r_V);
        WriteElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
      }

    });
  });

}

// kernel void InterpTranspose(const CeedInt num_elem, global const CeedScalar *restrict d_interp_1d, global const CeedScalar *restrict d_U,
//                             global CeedScalar *restrict d_V) {
//   // local size:
//   // 1d: elems_per_block * T_1d
//   // 2d,3d: elems_per_block * T_1d * T_1d
//   local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
//  private
//   CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
//  private
//   CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];

//   local CeedScalar  scratch[BASIS_INTERP_SCRATCH_SIZE];
//   local CeedScalar *elem_scratch = scratch + get_local_id(2) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

//   loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
//   work_group_barrier(CLK_LOCAL_MEM_FENCE);

//   if (BASIS_DIM == 1) {
//     ReadElementStrided1d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
//     InterpTranspose1d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
//     WriteElementStrided1d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);

//   } else if (BASIS_DIM == 2) {
//     ReadElementStrided2d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
//     InterpTransposeTensor2d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
//     WriteElementStrided2d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);

//   } else if (BASIS_DIM == 3) {
//     ReadElementStrided3d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
//     InterpTransposeTensor3d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, r_V, elem_scratch);
//     WriteElementStrided3d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
//   }
// }

extern "C" void InterpTranspose(sycl::queue &sycl_queue, sycl::nd_range<3> kernel_range, const CeedInt num_elem, const CeedScalar *__restrict__ d_interp_1d, const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
                                  
  std::vector<sycl::event> e;
  if (!sycl_queue.is_in_order()) e = {sycl_queue.ext_oneapi_submit_barrier()};
  
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    
    sycl::local_accessor<CeedScalar> s_memS(BASIS_INTERP_SCRATCH_SIZE, cgh);
    sycl::local_accessor<CeedScalar> s_memB(BASIS_P_1D * BASIS_Q_1D, cgh);
    
    cgh.parallel_for<CeedSyclSharedBasis_InterpTranspose<BASIS_DIM,BASIS_P_1D,BASIS_Q_1D>>(kernel_range, [=](sycl::nd_item<3> item) {     
      CeedScalar *scratch = s_memS.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_B     = s_memB.get_multi_ptr<sycl::access::decorated::yes>().get();

      SharedData_Sycl data;
      // data.work_item = item;
      data.item_id_x = item.get_local_id(2);
      data.item_id_y = item.get_local_id(1);
      data.item_id_z = item.get_global_id(0);
      data.item_id   = item.get_local_linear_id();
      data.group_size = item.get_local_range(0) * item.get_local_range(1) * item.get_local_range(2);
      data.scratch = scratch + item.get_local_id(0) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);
      
      CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
      CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
      
      // load interp_1d into shared memory
      loadMatrix<BASIS_P_1D, BASIS_Q_1D> (data, d_interp_1d, s_B);
      item.barrier(sycl::access::fence_space::local_space);
      // sycl::group_barrier(item.get_group());
      
      if (BASIS_DIM == 1) {
        ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1,  BASIS_NUM_QPTS * num_elem,  BASIS_NUM_QPTS, d_U, r_U);
        InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, r_V);
        WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      } else if (BASIS_DIM == 2) {
        ReadElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1,  BASIS_NUM_QPTS * num_elem,  BASIS_NUM_QPTS, d_U, r_U);
        InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, r_V);
        WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      } else if (BASIS_DIM == 3) {
        ReadElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1,  BASIS_NUM_QPTS * num_elem,  BASIS_NUM_QPTS, d_U, r_U);
        InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, r_V);
        WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      }
    });
  });

}

extern "C" void InterpTransposeAdd(sycl::queue &sycl_queue, sycl::nd_range<3> kernel_range, const CeedInt num_elem, const CeedScalar *__restrict__ d_interp_1d, const CeedScalar *__restrict__ d_U,
                                   CeedScalar *__restrict__ d_V) {
    
  std::vector<sycl::event> e;
  if (!sycl_queue.is_in_order()) e = {sycl_queue.ext_oneapi_submit_barrier()};
  
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    
    sycl::local_accessor<CeedScalar> s_memS(BASIS_INTERP_SCRATCH_SIZE, cgh);
    sycl::local_accessor<CeedScalar> s_memB(BASIS_P_1D * BASIS_Q_1D, cgh);
    
    cgh.parallel_for<CeedSyclSharedBasis_InterpTransposeAdd<BASIS_DIM,BASIS_P_1D,BASIS_Q_1D>>(kernel_range, [=](sycl::nd_item<3> item) {
      CeedScalar *scratch = s_memS.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_B     = s_memB.get_multi_ptr<sycl::access::decorated::yes>().get();

      SharedData_Sycl data;
      // data.work_item = item;
      data.item_id_x = item.get_local_id(2);
      data.item_id_y = item.get_local_id(1);
      data.item_id_z = item.get_global_id(0);
      data.item_id   = item.get_local_linear_id();
      data.group_size = item.get_local_range(0) * item.get_local_range(1) * item.get_local_range(2);
      data.scratch = scratch + item.get_local_id(0) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);
      
      CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
      CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
      
      // load interp_1d into shared memory
      loadMatrix<BASIS_P_1D, BASIS_Q_1D> (data, d_interp_1d, s_B);
      item.barrier(sycl::access::fence_space::local_space);
      // sycl::group_barrier(item.get_group());
      
      if (BASIS_DIM == 1) {
        ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1,  BASIS_NUM_QPTS * num_elem,  BASIS_NUM_QPTS, d_U, r_U);
        InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, r_V);
        SumElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      } else if (BASIS_DIM == 2) {
        ReadElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1,  BASIS_NUM_QPTS * num_elem,  BASIS_NUM_QPTS, d_U, r_U);
        InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, r_V);
        SumElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      } else if (BASIS_DIM == 3) {
        ReadElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1,  BASIS_NUM_QPTS * num_elem,  BASIS_NUM_QPTS, d_U, r_U);
        InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, r_V);
        SumElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      }
    });
  });

}

//------------------------------------------------------------------------------
// Grad kernel by dim
//------------------------------------------------------------------------------
// kernel void Grad(const CeedInt num_elem, global const CeedScalar *restrict d_interp_1d, global const CeedScalar *restrict d_grad_1d,
//                  global const CeedScalar *restrict d_U, global CeedScalar *restrict d_V) {
//   local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];  // Todo, don't allocate s_B for dimension 1
//   local CeedScalar s_G[BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D)];

//  private
//   CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
//  private
//   CeedScalar r_V[BASIS_NUM_COMP * BASIS_DIM * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

//   local CeedScalar  scratch[BASIS_GRAD_SCRATCH_SIZE];
//   local CeedScalar *elem_scratch = scratch + get_local_id(2) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

//   loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
//   loadMatrix(BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D), d_grad_1d, s_G);
//   work_group_barrier(CLK_LOCAL_MEM_FENCE);

//   if (BASIS_DIM == 1) {
//     ReadElementStrided1d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
//     Grad1d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_G, r_V, elem_scratch);
//     WriteElementStrided1d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);

//   } else if (BASIS_DIM == 2) {
//     ReadElementStrided2d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
//     GradTensor2d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
//     WriteElementStrided2d(BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);

//   } else if (BASIS_DIM == 3) {
//     ReadElementStrided3d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
//     if (BASIS_HAS_COLLOCATED_GRAD) GradTensorCollocated3d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
//     else GradTensor3d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
//     WriteElementStrided3d(BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
//   }
// }

extern "C" void Grad(sycl::queue &sycl_queue, sycl::nd_range<3> kernel_range, const CeedInt num_elem, const CeedScalar *__restrict__ d_interp_1d, const CeedScalar *__restrict__ d_grad_1d, 
                     const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  
  std::vector<sycl::event> e;
  if (!sycl_queue.is_in_order()) e = {sycl_queue.ext_oneapi_submit_barrier()};
  
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);

    sycl::local_accessor<CeedScalar> s_memS(BASIS_INTERP_SCRATCH_SIZE, cgh);
    sycl::local_accessor<CeedScalar> s_memB(BASIS_P_1D * BASIS_Q_1D, cgh); // UMESH: Todo, don't allocate s_B for dimension 1
    sycl::local_accessor<CeedScalar> s_memG(BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D), cgh);
    
    cgh.parallel_for<CeedSyclSharedBasis_Grad<BASIS_DIM,BASIS_P_1D,BASIS_Q_1D>>(kernel_range, [=](sycl::nd_item<3> item) {
      CeedScalar *scratch = s_memS.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_B     = s_memB.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_G     = s_memG.get_multi_ptr<sycl::access::decorated::yes>().get();

      SharedData_Sycl data;
      // data.work_item = item;
      data.item_id_x = item.get_local_id(2);
      data.item_id_y = item.get_local_id(1);
      data.item_id_z = item.get_global_id(0);
      data.item_id   = item.get_local_linear_id();
      data.group_size = item.get_local_range(0) * item.get_local_range(1) * item.get_local_range(2);
      data.scratch = scratch + item.get_local_id(0) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);
      
      CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
      CeedScalar r_V[BASIS_NUM_COMP * BASIS_DIM * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
      
      // load interp_1d and grad_1d into shared memory
      loadMatrix<BASIS_P_1D, BASIS_Q_1D> (data, d_interp_1d, s_B);
      loadMatrix<BASIS_Q_1D, BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D>(data, d_grad_1d, s_G);
      item.barrier(sycl::access::fence_space::local_space);
      
      if (BASIS_DIM == 1) {
        ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
        Grad1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_G, r_V);
        WriteElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
      } else if (BASIS_DIM == 2) {
        ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
        GradTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, s_G, r_V);
        WriteElementStrided2d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
      } else if (BASIS_DIM == 3) {
        ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, d_U, r_U);
        if (BASIS_HAS_COLLOCATED_GRAD) GradTensorCollocated3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, s_G, r_V);
        GradTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, s_G, r_V);
        WriteElementStrided3d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_V, d_V);
      }
    });
  });

}

// kernel void GradTranspose(const CeedInt num_elem, global const CeedScalar *restrict d_interp_1d, global const CeedScalar *restrict d_grad_1d,
//                           global const CeedScalar *restrict d_U, global CeedScalar *restrict d_V) {
//   local CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];  // Todo, don't allocate s_B for dimension 1
//   local CeedScalar s_G[BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D)];

//  private
//   CeedScalar r_U[BASIS_NUM_COMP * BASIS_DIM * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
//  private
//   CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];

//   local CeedScalar  scratch[BASIS_GRAD_SCRATCH_SIZE];
//   local CeedScalar *elem_scratch = scratch + get_local_id(2) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

//   loadMatrix(BASIS_P_1D * BASIS_Q_1D, d_interp_1d, s_B);
//   loadMatrix(BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D), d_grad_1d, s_G);
//   work_group_barrier(CLK_LOCAL_MEM_FENCE);

//   if (BASIS_DIM == 1) {
//     ReadElementStrided1d(BASIS_NUM_COMP, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
//     GradTranspose1d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_G, r_V, elem_scratch);
//     WriteElementStrided1d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);

//   } else if (BASIS_DIM == 2) {
//     ReadElementStrided2d(BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
//     GradTransposeTensor2d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
//     WriteElementStrided2d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);

//   } else if (BASIS_DIM == 3) {
//     ReadElementStrided3d(BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
//     if (BASIS_HAS_COLLOCATED_GRAD) GradTransposeTensorCollocated3d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
//     else GradTransposeTensor3d(BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, r_U, s_B, s_G, r_V, elem_scratch);
//     WriteElementStrided3d(BASIS_NUM_COMP, BASIS_P_1D, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
//   }
// }

extern "C" void GradTranspose(sycl::queue &sycl_queue, sycl::nd_range<3> kernel_range, const CeedInt num_elem, const CeedScalar *__restrict__ d_interp_1d, const CeedScalar *__restrict__ d_grad_1d,
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {

  std::vector<sycl::event> e;
  if (!sycl_queue.is_in_order()) e = {sycl_queue.ext_oneapi_submit_barrier()};
  
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    
    sycl::local_accessor<CeedScalar> s_memS(BASIS_INTERP_SCRATCH_SIZE, cgh);
    sycl::local_accessor<CeedScalar> s_memB(BASIS_P_1D * BASIS_Q_1D, cgh); // UMESH: Todo, don't allocate s_B for dimension 1
    sycl::local_accessor<CeedScalar> s_memG(BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D), cgh);
    
    cgh.parallel_for<CeedSyclSharedBasis_GradTranspose<BASIS_DIM,BASIS_P_1D,BASIS_Q_1D>>(kernel_range, [=](sycl::nd_item<3> item) {
      CeedScalar *scratch = s_memS.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_B     = s_memB.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_G     = s_memG.get_multi_ptr<sycl::access::decorated::yes>().get();

      SharedData_Sycl data;
      // data.work_item = item;
      data.item_id_x = item.get_local_id(2);
      data.item_id_y = item.get_local_id(1);
      data.item_id_z = item.get_global_id(0);
      data.item_id   = item.get_local_linear_id();
      data.group_size = item.get_local_range(0) * item.get_local_range(1) * item.get_local_range(2);
      data.scratch = scratch + item.get_local_id(0) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);
      
      CeedScalar r_U[BASIS_NUM_COMP * BASIS_DIM * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
      CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
      
      // load interp_1d and grad_1d into shared memory
      loadMatrix<BASIS_P_1D, BASIS_Q_1D> (data, d_interp_1d, s_B);
      loadMatrix<BASIS_Q_1D, BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D>(data, d_grad_1d, s_G);
      item.barrier(sycl::access::fence_space::local_space);
      
      if (BASIS_DIM == 1) {
        ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
        GradTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_G, r_V);
        WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      } else if (BASIS_DIM == 2) {
        ReadElementStrided2d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
        GradTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, s_G, r_V);
        WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      } else if (BASIS_DIM == 3) {
        ReadElementStrided3d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
        if (BASIS_HAS_COLLOCATED_GRAD) GradTransposeTensorCollocated3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, s_G, r_V);
        GradTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, s_G, r_V);
        WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      }
    });
  });

}

extern "C" void GradTransposeAdd(sycl::queue &sycl_queue, sycl::nd_range<3> kernel_range, const CeedInt num_elem, const CeedScalar *__restrict__ d_interp_1d, const CeedScalar *__restrict__ d_grad_1d,
                                 const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {

  std::vector<sycl::event> e;
  if (!sycl_queue.is_in_order()) e = {sycl_queue.ext_oneapi_submit_barrier()};
  
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    
    sycl::local_accessor<CeedScalar> s_memS(BASIS_INTERP_SCRATCH_SIZE, cgh);
    sycl::local_accessor<CeedScalar> s_memB(BASIS_P_1D * BASIS_Q_1D, cgh); // UMESH: Todo, don't allocate s_B for dimension 1
    sycl::local_accessor<CeedScalar> s_memG(BASIS_Q_1D * (BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D), cgh);
    
    cgh.parallel_for<CeedSyclSharedBasis_GradTransposeAdd<BASIS_DIM,BASIS_P_1D,BASIS_Q_1D>>(kernel_range, [=](sycl::nd_item<3> item) {
      CeedScalar *scratch = s_memS.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_B     = s_memB.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_G     = s_memG.get_multi_ptr<sycl::access::decorated::yes>().get();

      SharedData_Sycl data;
      // data.work_item = item;
      data.item_id_x = item.get_local_id(2);
      data.item_id_y = item.get_local_id(1);
      data.item_id_z = item.get_global_id(0);
      data.item_id   = item.get_local_linear_id();
      data.group_size = item.get_local_range(0) * item.get_local_range(1) * item.get_local_range(2);
      data.scratch = scratch + item.get_local_id(0) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);
      
      CeedScalar r_U[BASIS_NUM_COMP * BASIS_DIM * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
      CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
      
      // load interp_1d and grad_1d into shared memory
      loadMatrix<BASIS_P_1D, BASIS_Q_1D> (data, d_interp_1d, s_B);
      loadMatrix<BASIS_Q_1D, BASIS_HAS_COLLOCATED_GRAD ? BASIS_Q_1D : BASIS_P_1D>(data, d_grad_1d, s_G);
      item.barrier(sycl::access::fence_space::local_space);
      
      if (BASIS_DIM == 1) {
        ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
        GradTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_G, r_V);
        SumElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      } else if (BASIS_DIM == 2) {
        ReadElementStrided2d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
        GradTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, s_G, r_V);
        SumElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      } else if (BASIS_DIM == 3) {
        ReadElementStrided3d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, d_U, r_U);
        if (BASIS_HAS_COLLOCATED_GRAD) GradTransposeTensorCollocated3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, s_G, r_V);
        GradTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, item, r_U, s_B, s_G, r_V);
        SumElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, num_elem, 1, BASIS_NUM_NODES * num_elem, BASIS_NUM_NODES, r_V, d_V);
      }
    });
  });

}

//------------------------------------------------------------------------------
// Weight kernels by dim
//------------------------------------------------------------------------------
// kernel void Weight(const CeedInt num_elem, global const CeedScalar *restrict q_weight_1d, global CeedScalar *restrict d_W) {
//  private
//   CeedScalar r_W[BASIS_DIM > 2 ? BASIS_Q_1D : 1];

//   // void prefetch(q_weight_1d,BASIS_Q_1D);

//   if (BASIS_DIM == 1) {
//     Weight1d(BASIS_Q_1D, q_weight_1d, r_W);
//     WriteElementStrided1d(1, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_W, d_W);

//   } else if (BASIS_DIM == 2) {
//     WeightTensor2d(BASIS_Q_1D, q_weight_1d, r_W);
//     WriteElementStrided2d(1, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_W, d_W);

//   } else if (BASIS_DIM == 3) {
//     WeightTensor3d(BASIS_Q_1D, q_weight_1d, r_W);
//     WriteElementStrided3d(1, BASIS_Q_1D, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_W, d_W);
//   }
// }

extern "C" void Weight(sycl::queue &sycl_queue, sycl::nd_range<3> kernel_range, const CeedInt num_elem, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *__restrict__ d_W) {
  std::vector<sycl::event> e;
  if (!sycl_queue.is_in_order()) e = {sycl_queue.ext_oneapi_submit_barrier()};
  
  sycl_queue.parallel_for<CeedSyclSharedBasis_Weight<BASIS_DIM,BASIS_Q_1D>>(kernel_range, e, [=](sycl::nd_item<3> item) {
    SharedData_Sycl data;
    data.item_id_x = item.get_local_id(2);
    data.item_id_y = item.get_local_id(1);
    data.item_id_z = item.get_global_id(0);
    data.item_id   = item.get_local_linear_id();
    data.group_size = item.get_local_range(0) * item.get_local_range(1) * item.get_local_range(2);
    // data.scratch = scratch_WG + item.get_local_id(2) * T_1D * (BASIS_DIM > 1 ? T_1D : 1);
    
    CeedScalar r_W[BASIS_DIM > 2 ? BASIS_Q_1D : 1];
    
    if (BASIS_DIM == 1) {
      Weight1d<BASIS_Q_1D>(data, q_weight_1d, r_W);
      WriteElementStrided1d<1, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_W, d_W);
  
    } else if (BASIS_DIM == 2) {
      WeightTensor2d<BASIS_Q_1D>(data, q_weight_1d, r_W);
      WriteElementStrided2d<1, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_W, d_W);
  
    } else if (BASIS_DIM == 3) {
      WeightTensor3d<BASIS_Q_1D>(data, q_weight_1d, r_W);
      WriteElementStrided3d<1, BASIS_Q_1D>(data, num_elem, 1, BASIS_NUM_QPTS * num_elem, BASIS_NUM_QPTS, r_W, d_W);
    }
  });

}
