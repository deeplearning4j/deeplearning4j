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
// CUDA implementations of shape metadata op helpers.
//
// These use device-to-device operations so they are capturable by CUDA graphs.
// Without this fix, shape_of/size/rank/size_at write to HOST memory then
// syncToDevice (H2D copy). CUDA graph capture records the H2D copy but NOT
// the host-side write. On graph replay, the H2D copy transfers stale data,
// causing downstream ops (gather, etc.) to read zeros.
//
// Fix: Read shape metadata directly from device-side shapeInfo (specialShapeInfo)
// and write to device output buffer using D2D cudaMemcpyAsync or simple kernels.
// Both are fully capturable by CUDA graphs.
//

#include <ops/declarable/helpers/shapeOpsHelper.h>
#include <helpers/shape.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
// shape_of: copy rank dimension values from device shapeInfo to device output
//
// Always uses a kernel to read from specialShapeInfo and write to output.
// This is fully capturable by CUDA graphs and works for all output types.
///////////////////////////////////////////////////////////////////

template <typename T>
static SD_KERNEL void shapeOfCudaKernel(const LongType* shapeInfo, void* vz, int rank) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rank) {
    // shapeInfo layout: [rank, dim0, dim1, ..., stride0, stride1, ..., extra, ews, order]
    // shape::shapeOf equivalent: shapeInfo[idx + 1]
    reinterpret_cast<T*>(vz)[idx] = static_cast<T>(shapeInfo[idx + 1]);
  }
}

template <typename T>
static void shapeOfCudaLauncher(LaunchContext* context, const LongType* deviceShapeInfo,
                                void* deviceOutput, int rank) {
  auto stream = context->getCudaStream();
  shapeOfCudaKernel<T><<<1, 256, 0, *stream>>>(deviceShapeInfo, deviceOutput, rank);
}

void shapeOfHelper(LaunchContext* context, NDArray* x, NDArray* z, int rank) {
  NDArray::prepareSpecialUse({z}, {x});

  // Use kernel for all output types — reads directly from device specialShapeInfo
  BUILD_SINGLE_SELECTOR(z->dataType(), shapeOfCudaLauncher,
                        (context, x->specialShapeInfo(), z->specialBuffer(), rank),
                        SD_INTEGER_TYPES);

  NDArray::registerSpecialUse({z}, {x});
}

///////////////////////////////////////////////////////////////////
// rank: write input rank (scalar) to device output
///////////////////////////////////////////////////////////////////

template <typename T>
static SD_KERNEL void rankCudaKernel(const LongType* shapeInfo, void* vz) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    reinterpret_cast<T*>(vz)[0] = static_cast<T>(shape::rank(shapeInfo));
  }
}

template <typename T>
static void rankCudaLauncher(LaunchContext* context, const LongType* deviceShapeInfo,
                             void* deviceOutput) {
  auto stream = context->getCudaStream();
  rankCudaKernel<T><<<1, 1, 0, *stream>>>(deviceShapeInfo, deviceOutput);
}

void rankHelper(LaunchContext* context, NDArray* input, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input});

  BUILD_SINGLE_SELECTOR(output->dataType(), rankCudaLauncher,
                        (context, input->specialShapeInfo(), output->specialBuffer()),
                        SD_COMMON_TYPES);

  NDArray::registerSpecialUse({output}, {input});
}

///////////////////////////////////////////////////////////////////
// size: write total element count (scalar) to device output
// shape::length() is SD_HOST_DEVICE, callable from CUDA kernels
///////////////////////////////////////////////////////////////////

template <typename T>
static SD_KERNEL void sizeCudaKernel(const LongType* shapeInfo, void* vz) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    reinterpret_cast<T*>(vz)[0] = static_cast<T>(shape::length(shapeInfo));
  }
}

template <typename T>
static void sizeCudaLauncher(LaunchContext* context, const LongType* deviceShapeInfo,
                             void* deviceOutput) {
  auto stream = context->getCudaStream();
  sizeCudaKernel<T><<<1, 1, 0, *stream>>>(deviceShapeInfo, deviceOutput);
}

void sizeHelper(LaunchContext* context, NDArray* input, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input});

  BUILD_SINGLE_SELECTOR(output->dataType(), sizeCudaLauncher,
                        (context, input->specialShapeInfo(), output->specialBuffer()),
                        SD_COMMON_TYPES);

  NDArray::registerSpecialUse({output}, {input});
}

///////////////////////////////////////////////////////////////////
// size_at: write size of a specific dimension (scalar) to device output
///////////////////////////////////////////////////////////////////

template <typename T>
static SD_KERNEL void sizeAtCudaKernel(const LongType* shapeInfo, void* vz, int dim) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // dim is already adjusted for negative values by the caller
    reinterpret_cast<T*>(vz)[0] = static_cast<T>(shapeInfo[dim + 1]);
  }
}

template <typename T>
static void sizeAtCudaLauncher(LaunchContext* context, const LongType* deviceShapeInfo,
                               void* deviceOutput, int dim) {
  auto stream = context->getCudaStream();
  sizeAtCudaKernel<T><<<1, 1, 0, *stream>>>(deviceShapeInfo, deviceOutput, dim);
}

void sizeAtHelper(LaunchContext* context, NDArray* input, NDArray* output, int dim) {
  NDArray::prepareSpecialUse({output}, {input});

  BUILD_SINGLE_SELECTOR(output->dataType(), sizeAtCudaLauncher,
                        (context, input->specialShapeInfo(), output->specialBuffer(), dim),
                        SD_INTEGER_TYPES);

  NDArray::registerSpecialUse({output}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
