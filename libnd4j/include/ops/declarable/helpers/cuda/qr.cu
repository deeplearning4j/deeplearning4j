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
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/qr.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {

// Fused Modified Gram-Schmidt QR kernel.
// One block per matrix. Threads parallelize over rows within each column step.
// Shared memory used for parallel reductions (norm, dot product).
// Work buffer in global memory (C-contiguous [M x N]).
template <typename T>
static SD_KERNEL void qrModifiedGramSchmidtKernel(
    const T* __restrict__ input, const LongType* __restrict__ inputShape,
    T* __restrict__ qBuf, const LongType* __restrict__ qShape,
    T* __restrict__ rBuf, const LongType* __restrict__ rShape,
    T* __restrict__ work,
    LongType M, LongType N, LongType K, LongType Qcols,
    bool fullMatrices) {

  extern __shared__ char sharedMem[];
  T* sdata = reinterpret_cast<T*>(sharedMem);

  const int tid = threadIdx.x;
  const int numThreads = blockDim.x;

  // Strides from device shape info
  const LongType inS0 = shape::stride(inputShape)[0];
  const LongType inS1 = shape::stride(inputShape)[1];
  const LongType qS0 = shape::stride(qShape)[0];
  const LongType qS1 = shape::stride(qShape)[1];
  const LongType rS0 = shape::stride(rShape)[0];
  const LongType rS1 = shape::stride(rShape)[1];

  // Copy input to C-contiguous work buffer using input strides
  for (LongType idx = tid; idx < M * N; idx += numThreads) {
    LongType i = idx / N;
    LongType j = idx % N;
    work[idx] = input[i * inS0 + j * inS1];
  }

  // Zero Q
  for (LongType idx = tid; idx < M * Qcols; idx += numThreads) {
    LongType i = idx / Qcols;
    LongType j = idx % Qcols;
    qBuf[i * qS0 + j * qS1] = T(0);
  }

  // Zero R
  for (LongType idx = tid; idx < K * N; idx += numThreads) {
    LongType i = idx / N;
    LongType j = idx % N;
    rBuf[i * rS0 + j * rS1] = T(0);
  }
  __syncthreads();

  // Modified Gram-Schmidt: sequential over columns, parallel over rows
  for (LongType j = 0; j < K; j++) {
    // 1. Compute ||work[:,j]||^2 via parallel reduction
    T partial = T(0);
    for (LongType i = tid; i < M; i += numThreads) {
      T v = work[i * N + j];
      partial += v * v;
    }
    sdata[tid] = partial;
    __syncthreads();

    // Tree reduction (blockDim.x must be power of 2, or we guard with bounds check)
    for (unsigned int s = numThreads / 2; s > 0; s >>= 1) {
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }

    T norm = math::sd_sqrt<T, T>(sdata[0]);

    // Skip near-zero columns (linearly dependent)
    if (norm < T(1e-10)) {
      continue;
    }

    // 2. Q[:,j] = work[:,j] / norm
    for (LongType i = tid; i < M; i += numThreads) {
      qBuf[i * qS0 + j * qS1] = work[i * N + j] / norm;
    }

    // 3. R[j,j] = norm (single thread)
    if (tid == 0) {
      rBuf[j * rS0 + j * rS1] = norm;
    }
    __syncthreads();

    // 4. Orthogonalize remaining columns against Q[:,j]
    for (LongType k = j + 1; k < N; k++) {
      // Dot product: Q[:,j]^T * work[:,k]
      T partialDot = T(0);
      for (LongType i = tid; i < M; i += numThreads) {
        partialDot += qBuf[i * qS0 + j * qS1] * work[i * N + k];
      }
      sdata[tid] = partialDot;
      __syncthreads();

      for (unsigned int s = numThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
          sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
      }

      T dot = sdata[0];

      // R[j,k] = dot (single thread)
      if (tid == 0) {
        rBuf[j * rS0 + k * rS1] = dot;
      }

      // work[:,k] -= dot * Q[:,j]
      for (LongType i = tid; i < M; i += numThreads) {
        work[i * N + k] -= dot * qBuf[i * qS0 + j * qS1];
      }
      __syncthreads();
    }
  }

  // Full matrices: identity columns for Q beyond K
  if (fullMatrices && M > K) {
    for (LongType j = K + tid; j < M; j += numThreads) {
      qBuf[j * qS0 + j * qS1] = T(1);
    }
  }
}

template <typename T>
void qrSingle(LaunchContext* context, NDArray* matrix, NDArray* Q, NDArray* R, bool const fullMatrices) {
  LongType M = matrix->sizeAt(0);
  LongType N = matrix->sizeAt(1);
  LongType K = std::min(M, N);
  LongType Qcols = Q->sizeAt(1);

  NDArray::prepareSpecialUse({Q, R}, {matrix});

  auto stream = context->getCudaStream();

  // Allocate C-contiguous work buffer on device
  // During CUDA graph capture, synchronous calls are illegal.
  T* work = nullptr;
  cudaError_t err;
  if (tl_graphExecutionActive) {
    err = cudaMallocAsync(&work, M * N * sizeof(T), *stream);
  } else {
    err = cudaMalloc(&work, M * N * sizeof(T));
  }
  if (err != cudaSuccess) {
    THROW_EXCEPTION("qr: Failed to allocate work buffer on device");
  }
  // Power-of-2 block size for shared memory reductions
  const int blockSize = 256;
  const int sharedMem = blockSize * sizeof(T);

  qrModifiedGramSchmidtKernel<T><<<1, blockSize, sharedMem, *stream>>>(
      reinterpret_cast<const T*>(matrix->specialBuffer()), matrix->specialShapeInfo(),
      reinterpret_cast<T*>(Q->specialBuffer()), Q->specialShapeInfo(),
      reinterpret_cast<T*>(R->specialBuffer()), R->specialShapeInfo(),
      work, M, N, K, Qcols, fullMatrices);

  sd::DebugHelper::checkErrorCode(stream, "qrModifiedGramSchmidtKernel failed");

  // During CUDA graph capture, synchronous calls are illegal.
  if (tl_graphExecutionActive) {
    cudaFreeAsync(work, *stream);
  } else {
    cudaFree(work);
  }

  NDArray::registerSpecialUse({Q, R}, {matrix});
}
BUILD_SINGLE_TEMPLATE(void qrSingle, (LaunchContext* context, NDArray* matrix, NDArray* Q, NDArray* R, bool const fullMatrices), SD_FLOAT_TYPES);

template <typename T>
void qr_(LaunchContext* context, NDArray* input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
  // For 2D (unbatched) inputs, allTensorsAlongDimension({0,1}) produces rank-0 TADs
  // which causes sizeAt(1) to throw in qrSingle. Process the single matrix directly.
  if (input->rankOf() == 2) {
    qrSingle<T>(context, input, outputQ, outputR, fullMatricies);
    return;
  }

  LongType lastDim = input->rankOf() - 1;
  LongType preLastDim = input->rankOf() - 2;

  NDArray::prepareSpecialUse({outputQ, outputR}, {input});
  ResultSet listOutQ(outputQ->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  ResultSet listOutR(outputR->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  ResultSet listInput(input->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));

  for (LongType batch = 0; batch < listInput.size(); batch++) {
    qrSingle<T>(context, listInput.at(batch), listOutQ.at(batch), listOutR.at(batch), fullMatricies);
  }
  NDArray::registerSpecialUse({outputQ, outputR}, {input});
}
BUILD_SINGLE_TEMPLATE(void qr_, (LaunchContext* context, NDArray* input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies), SD_FLOAT_TYPES);

void qr(LaunchContext* context, NDArray* input, NDArray* outputQ, NDArray* outputR,
        bool const fullMatricies) {
  BUILD_SINGLE_SELECTOR(input->dataType(), qr_, (context, input, outputQ, outputR, fullMatricies), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
