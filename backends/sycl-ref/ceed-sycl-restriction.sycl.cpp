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
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
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
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
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
