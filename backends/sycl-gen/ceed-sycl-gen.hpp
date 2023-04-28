// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_sycl_gen_h
#define _ceed_sycl_gen_h

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <sycl/sycl_runtime.h>

#include "../sycl/ceed-sycl-common.h"

typedef struct {
  CeedInt       dim;
  CeedInt       Q_1d;
  CeedInt       max_P_1d;
  syclModule_t   module;
  syclFunction_t op;
  FieldsInt_Sycl indices;
  Fields_Sycl    fields;
  Fields_Sycl    B;
  Fields_Sycl    G;
  CeedScalar   *W;
} CeedOperator_Sycl_gen;

typedef struct {
  char *q_function_name;
  char *q_function_source;
  void *d_c;
} CeedQFunction_Sycl_gen;

CEED_INTERN int CeedQFunctionCreate_Sycl_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Sycl_gen(CeedOperator op);

#endif  // _ceed_sycl_gen_h
