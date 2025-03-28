// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL type definitions
#ifndef CEED_SYCL_TYPES_H
#define CEED_SYCL_TYPES_H

#include <ceed/types.h>
#include <sycl/sycl.hpp>

#define CEED_SYCL_NUMBER_FIELDS 16

#ifdef __OPENCL_C_VERSION__
typedef struct {
  global const CeedScalar *inputs[CEED_SYCL_NUMBER_FIELDS];
  global CeedScalar       *outputs[CEED_SYCL_NUMBER_FIELDS];
} Fields_Sycl;

typedef struct {
  global const CeedInt *inputs[CEED_SYCL_NUMBER_FIELDS];
  global CeedInt       *outputs[CEED_SYCL_NUMBER_FIELDS];
} FieldsInt_Sycl;
#else
typedef struct {
  const CeedScalar *inputs[CEED_SYCL_NUMBER_FIELDS];
  CeedScalar       *outputs[CEED_SYCL_NUMBER_FIELDS];
} Fields_Sycl;

typedef struct {
  const CeedInt *inputs[CEED_SYCL_NUMBER_FIELDS];
  CeedInt       *outputs[CEED_SYCL_NUMBER_FIELDS];
} FieldsInt_Sycl;
#endif

typedef struct {
  // sycl::nd_item<3> work_item;
  CeedInt item_id_x;
  CeedInt item_id_y;
  CeedInt item_id_z;
  CeedInt item_id;
  CeedInt group_size;
  CeedScalar *scratch;
} SharedData_Sycl;

#endif  // CEED_SYCL_TYPES_H
