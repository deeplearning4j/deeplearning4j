/* ******************************************************************************
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

#include <ops/declarable/helpers/broadcastableFused.h>
#include <helpers/DebugHelper.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// CUDA kernel implementations for same-shape contiguous operations

template <typename T>
SD_KERNEL static void addContiguousKernel(const void* vx, const void* vy, void* vz, sd::LongType len) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (sd::LongType i = tid; i < len; i += step) {
    z[i] = x[i] + y[i];
  }
}

template <typename T>
SD_KERNEL static void subtractContiguousKernel(const void* vx, const void* vy, void* vz, sd::LongType len) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (sd::LongType i = tid; i < len; i += step) {
    z[i] = x[i] - y[i];
  }
}

template <typename T>
SD_KERNEL static void multiplyContiguousKernel(const void* vx, const void* vy, void* vz, sd::LongType len) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (sd::LongType i = tid; i < len; i += step) {
    z[i] = x[i] * y[i];
  }
}

template <typename T>
SD_KERNEL static void divideContiguousKernel(const void* vx, const void* vy, void* vz, sd::LongType len) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (sd::LongType i = tid; i < len; i += step) {
    z[i] = x[i] / y[i];
  }
}

//////////////////////////////////////////////////////////////////////////
// CUDA kernel implementations for 1D broadcast along last dimension

template <typename T>
SD_KERNEL static void add1DLastKernel(const void* vx, const void* vy, void* vz, sd::LongType numRows, sd::LongType lastDim) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  sd::LongType totalLen = numRows * lastDim;
  for (sd::LongType i = tid; i < totalLen; i += step) {
    sd::LongType col = i % lastDim;
    z[i] = x[i] + y[col];
  }
}

template <typename T>
SD_KERNEL static void multiply1DLastKernel(const void* vx, const void* vy, void* vz, sd::LongType numRows, sd::LongType lastDim) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  sd::LongType totalLen = numRows * lastDim;
  for (sd::LongType i = tid; i < totalLen; i += step) {
    sd::LongType col = i % lastDim;
    z[i] = x[i] * y[col];
  }
}

//////////////////////////////////////////////////////////////////////////
// Template launcher functions

template <typename T>
static void addContiguousLauncher(const void* vx, const void* vy, void* vz, sd::LongType len, cudaStream_t* stream) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid > 512) blocksPerGrid = 512;

  addContiguousKernel<T><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(vx, vy, vz, len);
}

template <typename T>
static void subtractContiguousLauncher(const void* vx, const void* vy, void* vz, sd::LongType len, cudaStream_t* stream) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid > 512) blocksPerGrid = 512;

  subtractContiguousKernel<T><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(vx, vy, vz, len);
}

template <typename T>
static void multiplyContiguousLauncher(const void* vx, const void* vy, void* vz, sd::LongType len, cudaStream_t* stream) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid > 512) blocksPerGrid = 512;

  multiplyContiguousKernel<T><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(vx, vy, vz, len);
}

template <typename T>
static void divideContiguousLauncher(const void* vx, const void* vy, void* vz, sd::LongType len, cudaStream_t* stream) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid > 512) blocksPerGrid = 512;

  divideContiguousKernel<T><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(vx, vy, vz, len);
}

template <typename T>
static void add1DLastLauncher(const void* vx, const void* vy, void* vz, sd::LongType numRows, sd::LongType lastDim, cudaStream_t* stream) {
  int threadsPerBlock = 256;
  sd::LongType totalLen = numRows * lastDim;
  int blocksPerGrid = (totalLen + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid > 512) blocksPerGrid = 512;

  add1DLastKernel<T><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(vx, vy, vz, numRows, lastDim);
}

template <typename T>
static void multiply1DLastLauncher(const void* vx, const void* vy, void* vz, sd::LongType numRows, sd::LongType lastDim, cudaStream_t* stream) {
  int threadsPerBlock = 256;
  sd::LongType totalLen = numRows * lastDim;
  int blocksPerGrid = (totalLen + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid > 512) blocksPerGrid = 512;

  multiply1DLastKernel<T><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(vx, vy, vz, numRows, lastDim);
}

//////////////////////////////////////////////////////////////////////////
// Public API implementations

void fusedAddContiguous(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType len = x.lengthOf();
  auto stream = x.getContext()->getCudaStream();

  BUILD_SINGLE_SELECTOR(x.dataType(), addContiguousLauncher,
    (x.specialBuffer(), y.specialBuffer(), z.specialBuffer(), len, stream), SD_COMMON_TYPES);

  sd::DebugHelper::checkErrorCode(stream, "fusedAddContiguous failed");
}

void fusedSubtractContiguous(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType len = x.lengthOf();
  auto stream = x.getContext()->getCudaStream();

  BUILD_SINGLE_SELECTOR(x.dataType(), subtractContiguousLauncher,
    (x.specialBuffer(), y.specialBuffer(), z.specialBuffer(), len, stream), SD_COMMON_TYPES);

  sd::DebugHelper::checkErrorCode(stream, "fusedSubtractContiguous failed");
}

void fusedMultiplyContiguous(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType len = x.lengthOf();
  auto stream = x.getContext()->getCudaStream();

  BUILD_SINGLE_SELECTOR(x.dataType(), multiplyContiguousLauncher,
    (x.specialBuffer(), y.specialBuffer(), z.specialBuffer(), len, stream), SD_COMMON_TYPES);

  sd::DebugHelper::checkErrorCode(stream, "fusedMultiplyContiguous failed");
}

void fusedDivideContiguous(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType len = x.lengthOf();
  auto stream = x.getContext()->getCudaStream();

  BUILD_SINGLE_SELECTOR(x.dataType(), divideContiguousLauncher,
    (x.specialBuffer(), y.specialBuffer(), z.specialBuffer(), len, stream), SD_COMMON_TYPES);

  sd::DebugHelper::checkErrorCode(stream, "fusedDivideContiguous failed");
}

void fusedAdd1DLast(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType lastDim = x.sizeAt(-1);
  const sd::LongType numRows = x.lengthOf() / lastDim;
  auto stream = x.getContext()->getCudaStream();

  BUILD_SINGLE_SELECTOR(x.dataType(), add1DLastLauncher,
    (x.specialBuffer(), y.specialBuffer(), z.specialBuffer(), numRows, lastDim, stream), SD_COMMON_TYPES);

  sd::DebugHelper::checkErrorCode(stream, "fusedAdd1DLast failed");
}

void fusedMultiply1DLast(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType lastDim = x.sizeAt(-1);
  const sd::LongType numRows = x.lengthOf() / lastDim;
  auto stream = x.getContext()->getCudaStream();

  BUILD_SINGLE_SELECTOR(x.dataType(), multiply1DLastLauncher,
    (x.specialBuffer(), y.specialBuffer(), z.specialBuffer(), numRows, lastDim, stream), SD_COMMON_TYPES);

  sd::DebugHelper::checkErrorCode(stream, "fusedMultiply1DLast failed");
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
