/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <ops/declarable/helpers/sparse_blas.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// csr_spmv — CPU
// y = A * x  (transposeA=0)  or  y = A^T * x  (transposeA=1)
// ---------------------------------------------------------------------------
template <typename X, typename I>
static void csrSpMV_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                     NDArray& x, NDArray& y,
                     sd::LongType rows, sd::LongType cols, int transposeA) {
  const I nrows = static_cast<I>(rows);

  const auto* vBuf  = values.bufferAsT<X>();
  const auto* ciBuf = colIdx.bufferAsT<I>();
  const auto* rpBuf = rowPtr.bufferAsT<I>();
  const auto* xBuf  = x.bufferAsT<X>();
  auto*       yBuf  = y.bufferAsT<X>();

  const LongType xStride0 = x.stridesOf()[0];
  const LongType yStride0 = y.stridesOf()[0];

  if (transposeA == 0) {
    // y[i] = sum_{k in [rowPtr[i], rowPtr[i+1])} values[k] * x[colIdx[k]]
    for (I i = 0; i < nrows; ++i) {
      X acc = static_cast<X>(0);
      const I start = rpBuf[i];
      const I end   = rpBuf[i + 1];
      for (I k = start; k < end; ++k) {
        acc += vBuf[k] * xBuf[static_cast<LongType>(ciBuf[k]) * xStride0];
      }
      yBuf[static_cast<LongType>(i) * yStride0] = acc;
    }
  } else {
    // y = A^T * x: y[colIdx[k]] += values[k] * x[i]  for row i
    // y is OUTPUT_NULLIFIED (pre-zeroed)
    for (I i = 0; i < nrows; ++i) {
      const I start = rpBuf[i];
      const I end   = rpBuf[i + 1];
      const X xi    = xBuf[static_cast<LongType>(i) * xStride0];
      for (I k = start; k < end; ++k) {
        yBuf[static_cast<LongType>(ciBuf[k]) * yStride0] += vBuf[k] * xi;
      }
    }
  }
}

void csr_spmv(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
              NDArray& x, NDArray& y,
              sd::LongType rows, sd::LongType cols, int transposeA) {
  NDArray::preparePrimaryUse({&y}, {&values, &colIdx, &rowPtr, &x});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMV_,
                        (values, colIdx, rowPtr, x, y, rows, cols, transposeA),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&y}, {&values, &colIdx, &rowPtr, &x});
}

// ---------------------------------------------------------------------------
// csr_spmm — CPU
// C = A * B  (transposeA=0)  or  C = A^T * B  (transposeA=1)
// A CSR [rows, cols], B dense [cols, n] (or [rows, n] when transposed)
// ---------------------------------------------------------------------------
template <typename X, typename I>
static void csrSpMM_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                     NDArray& B, NDArray& C,
                     sd::LongType rows, sd::LongType cols, int transposeA) {
  const I nrows = static_cast<I>(rows);
  const LongType n = B.sizeAt(1);

  const auto* vBuf  = values.bufferAsT<X>();
  const auto* ciBuf = colIdx.bufferAsT<I>();
  const auto* rpBuf = rowPtr.bufferAsT<I>();
  const auto* bBuf  = B.bufferAsT<X>();
  auto*       cBuf  = C.bufferAsT<X>();

  const LongType bStride0 = B.stridesOf()[0];
  const LongType bStride1 = B.stridesOf()[1];
  const LongType cStride0 = C.stridesOf()[0];
  const LongType cStride1 = C.stridesOf()[1];

  if (transposeA == 0) {
    // C[i, j] = sum_{k in row i} values[k] * B[colIdx[k], j]
    for (I i = 0; i < nrows; ++i) {
      const I start = rpBuf[i];
      const I end   = rpBuf[i + 1];
      for (LongType j = 0; j < n; ++j) {
        X acc = static_cast<X>(0);
        for (I k = start; k < end; ++k) {
          acc += vBuf[k] * bBuf[static_cast<LongType>(ciBuf[k]) * bStride0 + j * bStride1];
        }
        cBuf[static_cast<LongType>(i) * cStride0 + j * cStride1] = acc;
      }
    }
  } else {
    // C = A^T * B: C[colIdx[k], j] += values[k] * B[i, j]  for row i
    // C is OUTPUT_NULLIFIED (pre-zeroed)
    for (I i = 0; i < nrows; ++i) {
      const I start = rpBuf[i];
      const I end   = rpBuf[i + 1];
      for (I k = start; k < end; ++k) {
        const LongType ci = static_cast<LongType>(ciBuf[k]);
        for (LongType j = 0; j < n; ++j) {
          cBuf[ci * cStride0 + j * cStride1] +=
              vBuf[k] * bBuf[static_cast<LongType>(i) * bStride0 + j * bStride1];
        }
      }
    }
  }
}

void csr_spmm(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
              NDArray& B, NDArray& C,
              sd::LongType rows, sd::LongType cols, int transposeA) {
  NDArray::preparePrimaryUse({&C}, {&values, &colIdx, &rowPtr, &B});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMM_,
                        (values, colIdx, rowPtr, B, C, rows, cols, transposeA),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&C}, {&values, &colIdx, &rowPtr, &B});
}

// ---------------------------------------------------------------------------
// sddmm — CPU
// outValues[k] = sum_l D1[i, l] * D2[j, l]
// where i = row of nonzero k,  j = colIdx[k]
// ---------------------------------------------------------------------------
template <typename X, typename I>
static void sddmm_(NDArray& rowPtr, NDArray& colIdx,
                   NDArray& D1, NDArray& D2, NDArray& outValues,
                   sd::LongType rows) {
  const I nrows = static_cast<I>(rows);
  const LongType p = D1.sizeAt(1);

  const auto* rpBuf  = rowPtr.bufferAsT<I>();
  const auto* ciBuf  = colIdx.bufferAsT<I>();
  const auto* d1Buf  = D1.bufferAsT<X>();
  const auto* d2Buf  = D2.bufferAsT<X>();
  auto*       oBuf   = outValues.bufferAsT<X>();

  const LongType d1Stride0 = D1.stridesOf()[0];
  const LongType d1Stride1 = D1.stridesOf()[1];
  const LongType d2Stride0 = D2.stridesOf()[0];
  const LongType d2Stride1 = D2.stridesOf()[1];

  for (I i = 0; i < nrows; ++i) {
    const I start = rpBuf[i];
    const I end   = rpBuf[i + 1];
    for (I k = start; k < end; ++k) {
      const LongType j = static_cast<LongType>(ciBuf[k]);
      X acc = static_cast<X>(0);
      for (LongType l = 0; l < p; ++l) {
        acc += d1Buf[static_cast<LongType>(i) * d1Stride0 + l * d1Stride1]
             * d2Buf[j * d2Stride0 + l * d2Stride1];
      }
      oBuf[k] = acc;
    }
  }
}

void sddmm(NDArray& rowPtr, NDArray& colIdx,
           NDArray& D1, NDArray& D2, NDArray& outValues,
           sd::LongType rows, sd::LongType cols) {
  NDArray::preparePrimaryUse({&outValues}, {&rowPtr, &colIdx, &D1, &D2});

  BUILD_DOUBLE_SELECTOR(D1.dataType(), colIdx.dataType(), sddmm_,
                        (rowPtr, colIdx, D1, D2, outValues, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&outValues}, {&rowPtr, &colIdx, &D1, &D2});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
