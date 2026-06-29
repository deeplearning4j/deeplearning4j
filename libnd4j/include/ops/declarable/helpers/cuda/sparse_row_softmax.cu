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

//
// csr_row_softmax / csr_row_softmax_bp — CUDA implementation.
//
// Forward kernel:  one thread per row.  Reads values[rowPtr[i]..rowPtr[i+1])
//   via specialBuffer(); computes numerically-stable softmax in-row and writes
//   alpha.  No inter-thread communication; no atomics.
//
// Backward kernel: one thread per row.  Reads alpha, gradOut; computes
//   dot = sum_k alpha[k]*gradOut[k] and writes dValues[k] = alpha[k]*(gradOut[k]-dot).
//   No atomics; each thread owns its entire row.
//
// No raw cudaMalloc; no host-compute shim.
// prepareSpecialUse / registerSpecialUse bracket each kernel launch.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_row_softmax.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Forward kernel: one thread per row
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrRowSoftmaxKernel(
    const X* values,   // [nnz]    edge scores
    const I* rowPtr,   // [rows+1] row pointers
    X*       alpha,    // [nnz]    output softmax weights
    LongType rows)
{
  const LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows) return;

  const LongType eStart = static_cast<LongType>(rowPtr[i]);
  const LongType eEnd   = static_cast<LongType>(rowPtr[i + 1]);
  if (eStart == eEnd) return;  // empty row

  // Step 1: row maximum for numerical stability
  X m = values[eStart];
  for (LongType e = eStart + 1; e < eEnd; ++e) {
    X v = values[e];
    if (v > m) m = v;
  }

  // Step 2: shifted exp + partial sum
  X s = static_cast<X>(0);
  for (LongType e = eStart; e < eEnd; ++e) {
    X ex = sd::math::sd_exp<X, X>(values[e] - m);
    alpha[e] = ex;
    s += ex;
  }

  // Step 3: normalise
  for (LongType e = eStart; e < eEnd; ++e) {
    alpha[e] /= s;
  }
}

template <typename X, typename I>
static void csrRowSoftmaxCuda_(NDArray& values, NDArray& rowPtr, NDArray& alpha,
                                sd::LongType rows) {
  auto* stream = values.getContext()->getCudaStream();

  const X* vBuf  = reinterpret_cast<const X*>(values.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  X*       aBuf  = reinterpret_cast<X*>(alpha.specialBuffer());

  if (rows == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((rows + blockSize - 1) / blockSize);

  csrRowSoftmaxKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      vBuf, rpBuf, aBuf, rows);
}

void csr_row_softmax(NDArray& values, NDArray& rowPtr, NDArray& alpha,
                     sd::LongType rows) {
  NDArray::prepareSpecialUse({&alpha}, {&values, &rowPtr});

  BUILD_DOUBLE_SELECTOR(values.dataType(), rowPtr.dataType(), csrRowSoftmaxCuda_,
                        (values, rowPtr, alpha, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&alpha}, {&values, &rowPtr});
}

// ────────────────────────────────────────────────────────────────────────────
// Backward kernel: one thread per row
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrRowSoftmaxBpKernel(
    const X* alpha,    // [nnz]    forward output
    const I* rowPtr,   // [rows+1] row pointers
    const X* gradOut,  // [nnz]    upstream gradient
    X*       dValues,  // [nnz]    output gradient
    LongType rows)
{
  const LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows) return;

  const LongType eStart = static_cast<LongType>(rowPtr[i]);
  const LongType eEnd   = static_cast<LongType>(rowPtr[i + 1]);
  if (eStart == eEnd) return;  // empty row

  // dot_i = sum_k alpha[k] * gradOut[k]
  X dot = static_cast<X>(0);
  for (LongType e = eStart; e < eEnd; ++e) {
    dot += alpha[e] * gradOut[e];
  }

  // dValues[k] = alpha[k] * (gradOut[k] - dot_i)
  for (LongType e = eStart; e < eEnd; ++e) {
    dValues[e] = alpha[e] * (gradOut[e] - dot);
  }
}

template <typename X, typename I>
static void csrRowSoftmaxBpCuda_(NDArray& alpha, NDArray& rowPtr, NDArray& gradOut,
                                   NDArray& dValues, sd::LongType rows) {
  auto* stream = alpha.getContext()->getCudaStream();

  const X* aBuf  = reinterpret_cast<const X*>(alpha.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* goBuf = reinterpret_cast<const X*>(gradOut.specialBuffer());
  X*       dvBuf = reinterpret_cast<X*>(dValues.specialBuffer());

  if (rows == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((rows + blockSize - 1) / blockSize);

  csrRowSoftmaxBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      aBuf, rpBuf, goBuf, dvBuf, rows);
}

void csr_row_softmax_bp(NDArray& alpha, NDArray& rowPtr, NDArray& gradOut,
                         NDArray& dValues, sd::LongType rows) {
  NDArray::prepareSpecialUse({&dValues}, {&alpha, &rowPtr, &gradOut});

  BUILD_DOUBLE_SELECTOR(alpha.dataType(), rowPtr.dataType(), csrRowSoftmaxBpCuda_,
                        (alpha, rowPtr, gradOut, dValues, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dValues}, {&alpha, &rowPtr, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
