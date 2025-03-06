// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL backend QFunction read/write kernels
#include <ceed/types.h>

#define sqrt(x) sycl::sqrt(x)
#define  exp(x) sycl::exp(x)
#define  log(x) sycl::log(x)
#define  cos(x) sycl::cos(x)
#define  pow(x,y) sycl::pow(x,y)
#define log1p(x) sycl::log1p(x)

//------------------------------------------------------------------------------
// Read from quadrature points
//------------------------------------------------------------------------------
template <int SIZE>
inline void readQuads(CeedInt offset, CeedInt stride, const CeedScalar *src, CeedScalar *dest) {
  for (CeedInt i = 0; i < SIZE; ++i) dest[i] = src[offset + stride * i];
}

//------------------------------------------------------------------------------
// Write at quadrature points
//------------------------------------------------------------------------------
template <int SIZE>
inline void writeQuads(CeedInt offset, CeedInt stride, const CeedScalar *src, CeedScalar *dest) {
  for (CeedInt i = 0; i < SIZE; ++i) dest[offset + stride * i] = src[i];
}
