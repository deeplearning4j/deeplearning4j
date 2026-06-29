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
// csr_diag_mm_bp — CUDA implementation.
//
// One CUDA thread per stored nonzero entry e in [0, nnz).
//
//   Step 1 — binary-search aRowPtr[0..rows] for the unique row i satisfying
//             aRowPtr[i] <= e < aRowPtr[i+1].
//             (Mirrors csrDiagMmKernel in sparse_diag_mm.cu exactly.)
//
//   Step 2 — j = aColIdx[e]
//
//   Step 3 — t = gradOut[e] * aValues[e]
//
//   Step 4 — dAValues[e] = gradOut[e] * dl[i] * dr[j]
//             Direct write — one thread uniquely owns entry e, no atomic needed.
//
//   Step 5 — ddl[i] += t * dr[j]
//             Multiple entries e can share the same row i, so atomicAdd is required.
//
//   Step 6 — ddr[j] += t * dl[i]
//             Multiple entries e can share the same column j, so atomicAdd is required.
//
// ddl and ddr are zeroed via cudaMemsetAsync on the context stream before the kernel.
//
// No raw cudaMalloc; no host-side compute shim.
// prepareSpecialUse / registerSpecialUse bracket the kernel launch.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_diag_mm_bp.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Device kernel: one thread per nonzero entry e
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrDiagMmBpKernel(
    const X* aValues,   // [nnz]   — original values
    const I* aColIdx,   // [nnz]   — column indices
    const I* aRowPtr,   // [rows+1] — row pointers
    const X* dl,        // [rows]  — left diagonal
    const X* dr,        // [cols]  — right diagonal
    const X* gradOut,   // [nnz]   — upstream gradient
    X*       dAValues,  // [nnz]   — output: grad w.r.t. aValues (direct write)
    X*       ddl,       // [rows]  — output: grad w.r.t. dl (atomicAdd)
    X*       ddr,       // [cols]  — output: grad w.r.t. dr (atomicAdd)
    LongType nnz,
    LongType rows)
{
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;

  // ── Step 1: binary-search aRowPtr for the row i owning entry e ──
  LongType lo = 0, hi = rows - 1, i = 0;
  while (lo <= hi) {
    const LongType mid       = lo + (hi - lo) / 2;
    const LongType rpMid     = static_cast<LongType>(aRowPtr[mid]);
    const LongType rpMidNext = static_cast<LongType>(aRowPtr[mid + 1]);
    if (rpMid <= e && e < rpMidNext) {
      i = mid;
      break;
    } else if (rpMid > e) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }

  // ── Step 2: column of entry e ──
  const LongType j = static_cast<LongType>(aColIdx[e]);

  // ── Step 3: shared intermediate ──
  const X t = gradOut[e] * aValues[e];  // gradOut[e] * aValues[e]

  // ── Step 4: dAValues — direct write (e is unique per thread) ──
  dAValues[e] = gradOut[e] * dl[i] * dr[j];

  // ── Step 5: ddl[i] += t * dr[j]  (scatter over row i — atomicAdd) ──
  sd::math::atomics::sd_atomicAdd(&ddl[i], t * dr[j]);

  // ── Step 6: ddr[j] += t * dl[i]  (scatter over col j — atomicAdd) ──
  sd::math::atomics::sd_atomicAdd(&ddr[j], t * dl[i]);
}

// ────────────────────────────────────────────────────────────────────────────
// Device-dispatch wrapper (called via BUILD_DOUBLE_SELECTOR)
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrDiagMmBpCuda_(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                              NDArray& dl,      NDArray& dr,      NDArray& gradOut,
                              NDArray& dAValues, NDArray& ddl,    NDArray& ddr,
                              sd::LongType rows, sd::LongType /*cols*/) {
  auto* stream = aValues.getContext()->getCudaStream();

  const X* avBuf  = reinterpret_cast<const X*>(aValues.specialBuffer());
  const I* ciBuf  = reinterpret_cast<const I*>(aColIdx.specialBuffer());
  const I* rpBuf  = reinterpret_cast<const I*>(aRowPtr.specialBuffer());
  const X* dlBuf  = reinterpret_cast<const X*>(dl.specialBuffer());
  const X* drBuf  = reinterpret_cast<const X*>(dr.specialBuffer());
  const X* goBuf  = reinterpret_cast<const X*>(gradOut.specialBuffer());
  X*       daBuf  = reinterpret_cast<X*>(dAValues.specialBuffer());
  X*       ddlBuf = reinterpret_cast<X*>(ddl.specialBuffer());
  X*       ddrBuf = reinterpret_cast<X*>(ddr.specialBuffer());

  const LongType nnz = aValues.lengthOf();
  if (nnz == 0) return;

  // Zero the scatter-accumulation buffers on the same stream before the kernel
  cudaMemsetAsync(ddlBuf, 0, sizeof(X) * static_cast<size_t>(ddl.lengthOf()), *stream);
  cudaMemsetAsync(ddrBuf, 0, sizeof(X) * static_cast<size_t>(ddr.lengthOf()), *stream);

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  csrDiagMmBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      avBuf, ciBuf, rpBuf, dlBuf, drBuf, goBuf, daBuf, ddlBuf, ddrBuf, nnz, rows);
}

// ────────────────────────────────────────────────────────────────────────────
// Public entry point
// ────────────────────────────────────────────────────────────────────────────

void csr_diag_mm_bp(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                    NDArray& dl,      NDArray& dr,      NDArray& gradOut,
                    NDArray& dAValues, NDArray& ddl,    NDArray& ddr,
                    sd::LongType rows, sd::LongType cols) {
  NDArray::prepareSpecialUse({&dAValues, &ddl, &ddr},
                             {&aValues, &aColIdx, &aRowPtr, &dl, &dr, &gradOut});

  BUILD_DOUBLE_SELECTOR(aValues.dataType(), aColIdx.dataType(), csrDiagMmBpCuda_,
                        (aValues, aColIdx, aRowPtr, dl, dr, gradOut,
                         dAValues, ddl, ddr, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dAValues, &ddl, &ddr},
                              {&aValues, &aColIdx, &aRowPtr, &dl, &dr, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
