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
// csr_diag_mm — CUDA implementation.
//
// Kernel design: one CUDA thread per stored nonzero entry e in [0, nnz).
//
//   Step 1 — find the row i that owns entry e:
//             binary-search aRowPtr[0..rows] for the unique i satisfying
//             aRowPtr[i] <= e < aRowPtr[i+1].
//             (Mirrors csrSddmmSparseKernel in sparse_sddmm_sparse.cu exactly.)
//
//   Step 2 — j = aColIdx[e]
//
//   Step 3 — outValues[e] = dl[i] * aValues[e] * dr[j]  (no atomics)
//
// No raw cudaMalloc; no host-side compute shim; no atomics needed.
// prepareSpecialUse / registerSpecialUse bracket the kernel launch.
//

#include <cuda_runtime.h>
#include <ops/declarable/helpers/sparse_diag_mm.h>
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
static SD_KERNEL void csrDiagMmKernel(
    const X* aValues,   // [nnz]
    const I* aColIdx,   // [nnz]
    const I* aRowPtr,   // [rows+1]
    const X* dl,        // [rows]
    const X* dr,        // [cols]
    X*       outValues, // [nnz]
    LongType nnz,
    LongType rows)
{
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;

  // ── Step 1: binary-search aRowPtr for the row i owning entry e ──
  //
  // Invariant: aRowPtr[i] <= e < aRowPtr[i+1]
  LongType lo = 0, hi = rows - 1, i = 0;
  while (lo <= hi) {
    const LongType mid        = lo + (hi - lo) / 2;
    const LongType rpMid      = static_cast<LongType>(aRowPtr[mid]);
    const LongType rpMidNext  = static_cast<LongType>(aRowPtr[mid + 1]);
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

  // ── Step 3: scaled output — no atomics, each e is owned by exactly one thread ──
  outValues[e] = dl[i] * aValues[e] * dr[j];
}

// ────────────────────────────────────────────────────────────────────────────
// Device-dispatch wrapper (called via BUILD_DOUBLE_SELECTOR)
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrDiagMmCuda_(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                            NDArray& dl,      NDArray& dr,
                            NDArray& outValues,
                            sd::LongType rows, sd::LongType /*cols*/) {
  auto* stream = outValues.getContext()->getCudaStream();

  const X* avBuf = reinterpret_cast<const X*>(aValues.specialBuffer());
  const I* ciBuf = reinterpret_cast<const I*>(aColIdx.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(aRowPtr.specialBuffer());
  const X* dlBuf = reinterpret_cast<const X*>(dl.specialBuffer());
  const X* drBuf = reinterpret_cast<const X*>(dr.specialBuffer());
  X*       oBuf  = reinterpret_cast<X*>(outValues.specialBuffer());

  const LongType nnz = outValues.lengthOf();
  if (nnz == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  csrDiagMmKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      avBuf, ciBuf, rpBuf, dlBuf, drBuf, oBuf, nnz, rows);
}

// ────────────────────────────────────────────────────────────────────────────
// Public entry point
// ────────────────────────────────────────────────────────────────────────────

void csr_diag_mm(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                 NDArray& dl,      NDArray& dr,
                 NDArray& outValues,
                 sd::LongType rows, sd::LongType cols) {
  NDArray::prepareSpecialUse({&outValues},
                             {&aValues, &aColIdx, &aRowPtr, &dl, &dr});

  BUILD_DOUBLE_SELECTOR(aValues.dataType(), aColIdx.dataType(), csrDiagMmCuda_,
                        (aValues, aColIdx, aRowPtr, dl, dr, outValues, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&outValues},
                              {&aValues, &aColIdx, &aRowPtr, &dl, &dr});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
