// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed-impl.h>
#include <string.h>
#include "ceed-ref.h"

static int CeedVectorSetArray_Ref(CeedVector vec, CeedMemType mtype,
                                  CeedCopyMode cmode, CeedScalar *array) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedFree(&impl->array_allocated); CeedChk(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(vec->length, &impl->array_allocated); CeedChk(ierr);
    impl->array = impl->array_allocated;
    if (array) memcpy(impl->array, array, vec->length * sizeof(array[0]));
    break;
  case CEED_OWN_POINTER:
    impl->array_allocated = array;
    impl->array = array;
    break;
  case CEED_USE_POINTER:
    impl->array = array;
  }
  return 0;
}

static int CeedVectorGetArray_Ref(CeedVector vec, CeedMemType mtype,
                                  CeedScalar **array) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  if (!impl->array) { // Allocate if array is not yet allocated
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }
  *array = impl->array;
  return 0;
}

static int CeedVectorGetArrayRead_Ref(CeedVector vec, CeedMemType mtype,
                                      const CeedScalar **array) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  if (!impl->array) { // Allocate if array is not yet allocated
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }
  *array = impl->array;
  return 0;
}

static int CeedVectorRestoreArray_Ref(CeedVector vec, CeedScalar **array) {
  *array = NULL;
  return 0;
}

static int CeedVectorRestoreArrayRead_Ref(CeedVector vec,
    const CeedScalar **array) {
  *array = NULL;
  return 0;
}

static int CeedVectorDestroy_Ref(CeedVector vec) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  ierr = CeedFree(&impl->array_allocated); CeedChk(ierr);
  ierr = CeedFree(&vec->data); CeedChk(ierr);
  return 0;
}

int CeedVectorCreate_Ref(CeedInt n, CeedVector vec) {
  CeedVector_Ref *impl;
  int ierr;

  vec->SetArray = CeedVectorSetArray_Ref;
  vec->GetArray = CeedVectorGetArray_Ref;
  vec->GetArrayRead = CeedVectorGetArrayRead_Ref;
  vec->RestoreArray = CeedVectorRestoreArray_Ref;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Ref;
  vec->Destroy = CeedVectorDestroy_Ref;
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  vec->data = impl;
  return 0;
}