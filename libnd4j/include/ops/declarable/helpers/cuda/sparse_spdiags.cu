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
// spdiags — CUDA implementation.
//
// One CUDA thread per diagonal entry (threads 0..n-1) writes values/colIdx/rowPtr[i].
// Thread n writes the sentinel rowPtr[n] = n.
// Grid is sized to cover n+1 threads, so one kernel handles all outputs.
//
// No raw cudaMalloc; no host-side compute shim — all arithmetic is on-device.
// prepareSpecialUse / registerSpecialUse bracket the kernel launch.
//

#include <cuda_runtime.h>
#include <ops/declarable/helpers/sparse_spdiags.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Device kernel: one thread per diagonal index i in [0, n], covering n+1 threads.
//   i < n  → write values[i], colIdx[i], rowPtr[i]
//   i == n → write rowPtr[n] = n  (sentinel)
// ────────────────────────────────────────────────────────────────────────────

template <typename X>
static SD_KERNEL void spdiagsKernel(const X* diag, X* values, int32_t* colIdx, int32_t* rowPtr,
                                     LongType n) {
  const LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i > n) return;

  if (i < n) {
    values[i]  = diag[i];
    colIdx[i]  = static_cast<int32_t>(i);
    rowPtr[i]  = static_cast<int32_t>(i);
  }
  // Thread i==n writes the final sentinel.
  if (i == n) {
    rowPtr[n] = static_cast<int32_t>(n);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Device-dispatch wrapper (called via BUILD_SINGLE_SELECTOR)
// ────────────────────────────────────────────────────────────────────────────

template <typename X>
static void spdiagsCuda_(NDArray& diag, NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                          sd::LongType n) {
  if (n == 0) return;

  auto* stream = values.getContext()->getCudaStream();

  const X*    dBuf  = reinterpret_cast<const X*>(diag.specialBuffer());
  X*          vBuf  = reinterpret_cast<X*>(values.specialBuffer());
  int32_t*    ciBuf = reinterpret_cast<int32_t*>(colIdx.specialBuffer());
  int32_t*    rpBuf = reinterpret_cast<int32_t*>(rowPtr.specialBuffer());

  const int blockSize = 256;
  // n+1 threads: indices 0..n-1 write values/colIdx/rowPtr[i], index n writes rowPtr[n].
  const int gridSize  = static_cast<int>((n + 1 + blockSize - 1) / blockSize);

  spdiagsKernel<X><<<gridSize, blockSize, 0, *stream>>>(dBuf, vBuf, ciBuf, rpBuf, n);
}

// ────────────────────────────────────────────────────────────────────────────
// Public entry point
// ────────────────────────────────────────────────────────────────────────────

void spdiags(sd::LaunchContext* context, NDArray& diag, NDArray& values,
             NDArray& colIdx, NDArray& rowPtr, sd::LongType n) {
  NDArray::prepareSpecialUse({&values, &colIdx, &rowPtr}, {&diag});

  BUILD_SINGLE_SELECTOR(diag.dataType(), spdiagsCuda_,
                        (diag, values, colIdx, rowPtr, n),
                        SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({&values, &colIdx, &rowPtr}, {&diag});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
