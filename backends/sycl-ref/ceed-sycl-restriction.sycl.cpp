// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>

#include <string>
#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Sycl(CeedElemRestriction r, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionApplyBlock_Sycl(CeedElemRestriction r, CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                       CeedRequest *request) {
  // LCOV_EXCL_START
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Sycl(CeedElemRestriction r, CeedMemType m_type, const CeedInt **offsets) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedElemRestriction_Sycl *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  switch (m_type) {
    case CEED_MEM_HOST:
      *offsets = impl->h_ind;
      break;
    case CEED_MEM_DEVICE:
      *offsets = impl->d_ind;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Sycl(CeedElemRestriction r) {
  CeedElemRestriction_Sycl *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());
  CeedCallSycl(ceed, sycl::free(impl->d_ind_allocated , data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_t_offsets     , data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_t_indices     , data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_l_vec_indices , data->sycl_context));
  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create transpose offsets and indices
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffset_Sycl(const CeedElemRestriction r, const CeedInt *indices) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedElemRestriction_Sycl *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  CeedSize l_size;
  CeedInt num_elem, elem_size, num_comp;
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetLVectorSize(r, &l_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));

  // Count num_nodes
  bool *is_node;
  CeedCallBackend(CeedCalloc(l_size, &is_node));
  const CeedInt size_indices = num_elem * elem_size;
  for(CeedInt i = 0; i< size_indices ; ++i) is_node[indices[i]] = 1;
  CeedInt num_nodes = 0;
  for(CeedInt i = 0; i< l_size; ++i) num_nodes += is_node[i];
  impl->num_nodes = num_nodes;

  // L-vector offsets array
  CeedInt *ind_to_offset, *l_vec_indices;
  CeedCallBackend(CeedCalloc(l_size, &ind_to_offset));
  CeedCallBackend(CeedCalloc(num_nodes, &l_vec_indices));
  CeedInt j = 0;
  for (CeedInt i = 0; i<l_size;i++) {
    if (is_node[i]) {
      l_vec_indices[j] = i;
      ind_to_offset[i] = j++;
    }
  }
  CeedCallBackend(CeedFree(&is_node));

  // Compute transpose offsets and indices
  const CeedInt size_offsets = num_nodes + 1;
  CeedInt *t_offsets;
  CeedCallBackend(CeedCalloc(size_offsets, &t_offsets));
  CeedInt *t_indices;
  CeedCallBackend(CeedMalloc(size_indices, &t_indices));
  // Count node multiplicity
  for (CeedInt e = 0; e < num_elem; ++e) {
    for(CeedInt i = 0; i < elem_size; ++i) ++t_offsets[ind_to_offset[indices[elem_size * e + i]] + 1];
  }
  // Convert to running sum
  for (CeedInt i = 1; i < size_offsets; ++i) t_offsets[i] += t_offsets[i - 1];
  // List all E-vec indices associated with L-vec node
  for (CeedInt e = 0; e < num_elem; ++e) {
    for (CeedInt i = 0; i < elem_size; ++i) {
      const CeedInt lid = elem_size*e + i;
      const CeedInt gid = indices[lid];
      t_indices[t_offsets[ind_to_offset[gid]]++] = lid;
    }
  }
  // Reset running sum
  for (int i = size_offsets - 1; i > 0; --i) t_offsets[i] = t_offsets[i - 1];
  t_offsets[0] = 0;

  // Copy data to device
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  // -- L-vector indices
  CeedCallSycl(ceed, impl->d_l_vec_indices = sycl::malloc_device<CeedInt>(num_nodes, data->sycl_device, data->sycl_context));
  CeedCallSycl(ceed, data->sycl_queue.copy<CeedInt>(l_vec_indices, impl->d_l_vec_indices, num_nodes));
  // -- Transpose offsets
  CeedCallSycl(ceed, impl->d_t_offsets = sycl::malloc_device<CeedInt>(size_offsets, data->sycl_device, data->sycl_context));
  CeedCallSycl(ceed, data->sycl_queue.copy<CeedInt>(t_offsets, impl->d_t_offsets, size_offsets));
  // -- Transpose indices
  CeedCallSycl(ceed, impl->d_t_indices = sycl::malloc_device<CeedInt>(size_indices, data->sycl_device, data->sycl_context));
  CeedCallSycl(ceed, data->sycl_queue.copy<CeedInt>(t_indices, impl->d_t_indices,size_indices));
  // Wait for all copies to complete and handle exceptions
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());

  // Cleanup
  CeedCallBackend(CeedFree(&ind_to_offset));
  CeedCallBackend(CeedFree(&l_vec_indices));
  CeedCallBackend(CeedFree(&t_offsets));
  CeedCallBackend(CeedFree(&t_indices));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create restriction
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Sycl(CeedMemType m_type, CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedElemRestriction_Sycl *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedInt num_elem, num_comp, elem_size;
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  CeedInt size = num_elem * elem_size;
  CeedInt strides[3] = {1, size, elem_size};
  CeedInt comp_stride = 1;

  // Stride data
  bool is_strided;
  CeedCallBackend(CeedElemRestrictionIsStrided(r, &is_strided));
  if (is_strided) {
    bool has_backend_strides;
    CeedCallBackend(CeedElemRestrictionHasBackendStrides(r, &has_backend_strides));
    if (!has_backend_strides) {
      CeedCallBackend(CeedElemRestrictionGetStrides(r, &strides));
    }
  } else {
    CeedCallBackend(CeedElemRestrictionGetCompStride(r, &comp_stride));
  }

  impl->h_ind = NULL;
  impl->h_ind_allocated = NULL;
  impl->d_ind           = NULL;
  impl->d_ind_allocated = NULL;
  impl->d_t_indices     = NULL;
  impl->d_t_offsets     = NULL;
  impl->num_nodes       = size;
  CeedCallBackend(CeedElemRestrictionSetData(r, impl));
  CeedInt layout[3] = {1, elem_size * num_elem, elem_size};
  CeedCallBackend(CeedElemRestrictionSetELayout(r, layout));

  // Set up device indices/offset arrays
  if (m_type == CEED_MEM_HOST) {
    switch (copy_mode) {
      case CEED_OWN_POINTER:
        impl->h_ind_allocated = (CeedInt *)indices;
	impl->h_ind = (CeedInt *)indices;
	break;
      case CEED_USE_POINTER:
	impl->h_ind = (CeedInt *)indices;
	break;
      case CEED_COPY_VALUES:
	if (indices != NULL) {
	  CeedCallBackend(CeedMalloc(elem_size*num_elem, &impl->h_ind_allocated));
	  memcpy(impl->h_ind_allocated, indices, elem_size * num_elem * sizeof(CeedInt));
	  impl->h_ind = impl->h_ind_allocated;
	}
	break;
    }
    if (indices != NULL) {
      CeedCallSycl(ceed, impl->d_ind = sycl::malloc_device<CeedInt>(size, data->sycl_device, data->sycl_context));
      impl->d_ind_allocated = impl->d_ind; // We own the device memory
      // Copy from host to device
      sycl::event copy_event = data->sycl_queue.copy<CeedInt>(indices,impl->d_ind,size);
      // Wait for copy to finish and handle exceptions
      CeedCallSycl(ceed, copy_event.wait_and_throw());
      CeedCallBackend(CeedElemRestrictionOffset_Sycl(r, indices));
    }
  } else if (m_type == CEED_MEM_DEVICE) {
    switch (copy_mode) {
      case CEED_COPY_VALUES:
        if (indices != NULL) {
	  CeedCallSycl(ceed, impl->d_ind = sycl::malloc_device<CeedInt>(size, data->sycl_device, data->sycl_context));
	  impl->d_ind_allocated = impl->d_ind;  // We own the device memory
          // Copy from device to device
	  sycl::event copy_event = data->sycl_queue.copy<CeedInt>(indices,impl->d_ind,size);
	  // Wait for copy to finish and handle exceptions
	  CeedCallSycl(ceed, copy_event.wait_and_throw());
	}
	break;
      case CEED_OWN_POINTER:
	impl->d_ind = (CeedInt *)indices;
	impl->d_ind_allocated = impl->d_ind;
	break;
      case CEED_USE_POINTER:
	impl->d_ind = (CeedInt *)indices;
    }
    if (indices !=NULL) {
      CeedCallBackend(CeedMalloc(elem_size * num_elem, &impl->h_ind_allocated));
      // Copy from device to host
      sycl::event copy_event = data->sycl_queue.copy<CeedInt>(impl->d_ind,impl->h_ind_allocated,elem_size*num_elem);
      CeedCallSycl(ceed, copy_event.wait_and_throw());
      impl->h_ind = impl->h_ind_allocated;
      CeedCallBackend(CeedElemRestrictionOffset_Sycl(r,indices));
    }
  } else {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST or DEVICE supported");
    // LCOV_EXCL_STOP
  }

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "ElemRestriction", r, "Apply", CeedElemRestrictionApply_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "ElemRestriction", r, "ApplyBlock", CeedElemRestrictionApplyBlock_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "ElemRestriction", r, "GetOffsets", CeedElemRestrictionGetOffsets_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "ElemRestriction", r, "Destroy", CeedElemRestrictionDestroy_Sycl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionCreateBlocked_Sycl(const CeedMemType m_type, const CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement blocked restrictions");
}
//------------------------------------------------------------------------------
