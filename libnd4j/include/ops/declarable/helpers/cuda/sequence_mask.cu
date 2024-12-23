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
//  @author GS <sgazeos@gmail.com>
//
#include <execution/cuda/LaunchDims.h>
#include <ops/declarable/helpers/sequence_mask.h>


#include "helpers/DebugHelper.h"
namespace sd {
namespace ops {
namespace helpers {

template <typename I, typename B>
__global__ static void sequenceMaskKernel(const void* inputBuf, const LongType* inputShape, void* pVz,
                                          const LongType* zTadShapeInfo, const LongType axis, int maxIndex) {
  // Reinterpret input and output buffers
  const I* input = reinterpret_cast<const I*>(inputBuf);
  B* output = reinterpret_cast<B*>(pVz);

  // Shared memory for caching shape information and related variables
  extern __shared__ unsigned char shmem[];
  // Pointers within shared memory
  LongType* sharedMem = reinterpret_cast<LongType*>(shmem);

  // Shared variables
  __shared__ LongType shared_inputLen;
  __shared__ LongType shared_outputLen;
  __shared__ int shared_inputRank;
  __shared__ int shared_zTadRank;
  __shared__ LongType shared_zDim;
  __shared__ LongType shared_totalThreads;

  // Cached shape and stride pointers
  __shared__ const LongType* shared_inputShape;
  __shared__ const LongType* shared_inputStride;
  __shared__ const LongType* shared_zTadShape;
  __shared__ const LongType* shared_zTadStride;

  if (threadIdx.x == 0) {
    // Cache input tensor shape and stride
    shared_inputRank = shape::rank(inputShape);
    shared_inputShape = shape::shapeOf(inputShape);
    shared_inputStride = shape::stride(inputShape);

    // Cache zTad tensor shape and stride
    shared_zTadRank = shape::rank(zTadShapeInfo);
    shared_zTadShape = shape::shapeOf(zTadShapeInfo);
    shared_zTadStride = shape::stride(zTadShapeInfo);
    shared_zDim = shared_zTadShape[axis]; // Assuming zDim is constant across splits

    // Cache lengths
    shared_inputLen = shape::length(inputShape);
    shared_outputLen = shape::length(zTadShapeInfo); // Assuming output tensors have the same shape

    // Calculate total threads across all blocks
    shared_totalThreads = gridDim.x * blockDim.x;
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  // Calculate the global thread ID
  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate space in shared memory for coordinates
  LongType* coords = sharedMem + threadIdx.x * shared_inputRank;

  // Loop over each index along the axis to apply the mask
  for (LongType i = tid; i < maxIndex; i += shared_totalThreads) {
    // Inner loop over all elements in the input tensor
    for (LongType k = threadIdx.x; k < shared_inputLen; k += blockDim.x) {
      // Convert linear index 'k' to multi-dimensional coordinates
      INDEX2COORDS(k, shared_inputRank, shared_inputShape, coords);

      LongType inputOffset;
      // Convert coordinates to linear index for input tensor
      COORDS2INDEX(shared_inputRank, shared_inputStride, coords, inputOffset);

      // Determine the split index along the specified axis
      LongType splitIndex = coords[axis] / shared_zDim;

      // Retrieve the pointer to the target output tensor based on splitIndex
      B* z = reinterpret_cast<B*>(reinterpret_cast<void**>(pVz)[splitIndex]);

      // Update the coordinate along the split axis
      coords[axis] %= shared_zDim;

      LongType zOffset;
      // Convert updated coordinates to linear index for z tensor
      COORDS2INDEX(shared_zTadRank, shared_zTadStride, coords, zOffset);

      // Apply the mask condition
      if (i < static_cast<LongType>(input[inputOffset])) {
        z[zOffset] = static_cast<B>(true);
      } else {
        z[zOffset] = static_cast<B>(false); // Optionally handle the false case
      }
    }
  }
}

template <typename I, typename B>
static void sequenceMask_(LaunchContext* context, NDArray* input, NDArray* output, int maxIndex) {
  dim3 launchDims = getSequenceMaskLaunchDims(maxIndex,*input);
  NDArray::prepareSpecialUse({output}, {input});
  auto stream = context->getCudaStream();
  sequenceMaskKernel<I, B><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
      input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), maxIndex);
  sd::DebugHelper::checkErrorCode(stream, "sequenceMaskKernel failed");

  NDArray::registerSpecialUse({output}, {input});
}

void sequenceMask(LaunchContext* context, NDArray* input, NDArray* output, int maxIndex) {
  BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), sequenceMask_, (context, input, output, maxIndex),
                        SD_INTEGER_TYPES, SD_COMMON_TYPES_EXTENDED);
}

BUILD_DOUBLE_TEMPLATE(template void sequenceMask_,
                      (sd::LaunchContext * context, NDArray* input, NDArray* output, int maxIndex), SD_INTEGER_TYPES,
                      SD_COMMON_TYPES_EXTENDED);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
