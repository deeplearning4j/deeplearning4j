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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 15.11.2018
//
#include <loops/special_kernels.h>

namespace sd {
template <typename T>
SD_DEVICE void concatKernelHStack(int numArrays, Pointer *data, Pointer *inputShapeInfos, void *vz,
                                  LongType *zShapeInfo) {
  // We expect each input array to be a vector, and the result (z) to be a 2D matrix for horizontal stacking.
  // The row dimension is presumably 1 for each input, with the column dimension the length of each vector,
  // stacked horizontally.

  auto z = reinterpret_cast<T*>(vz);
  auto inputData = reinterpret_cast<T**>(data);
  auto shapes = reinterpret_cast<LongType**>(inputShapeInfos);

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Output shape data in shared memory
  __shared__ int zRank;
  __shared__ const LongType* zShape;
  __shared__ const LongType* zStride;

  // Current input array shape data in shared memory
  __shared__ int inRank;
  __shared__ const LongType* inShape;
  __shared__ const LongType* inStride;

  // Working variables in shared memory
  __shared__ int inputLength;
  __shared__ int baseIdx;

  // Initialize output shape data once
  if (threadIdx.x == 0) {
    zRank = shape::rank(zShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  // Loop over all input arrays
  for (int r = blockIdx.x; r < numArrays; r += gridDim.x) {
    // Cache the current input array's shape data and compute offsets
    if (threadIdx.x == 0) {
      // Cache current input shape data
      inRank = shape::rank(shapes[r]);
      inShape = shape::shapeOf(shapes[r]);
      inStride = shape::stride(shapes[r]);

      // Compute base offset
      baseIdx = 0;
      for (int f = 0; f < r; f++) {
        baseIdx += shape::length(shapes[f]);
      }
      inputLength = shape::length(shapes[r]);
    }
    __syncthreads();

    // Each thread will copy a subset of data
    for (int i = tid; i < inputLength; i += blockDim.x * gridDim.x) {
      // Coordinates in the input vector
      LongType inCoords[SD_MAX_RANK];
      // Coordinates in the output 2D shape
      LongType outCoords[SD_MAX_RANK];

      // 1) Get input coordinates using cached shape data
      INDEX2COORDS(i, inRank, inShape, inCoords);
      LongType inOffset;
      COORDS2INDEX(inRank, inStride, inCoords, inOffset);

      // 2) The output coordinate index is baseIdx + i in the horizontal dimension
      const LongType outIndex = baseIdx + i;

      // Get output coordinates using cached shape data
      INDEX2COORDS(outIndex, zRank, zShape, outCoords);
      LongType outOffset;
      COORDS2INDEX(zRank, zStride, outCoords, outOffset);

      z[outOffset] = inputData[r][inOffset];
    }
    __syncthreads();
  }
}

template <typename T>
SD_KERNEL void execConcatKernelHStack(int numArrays, Pointer *data, Pointer *inputShapeInfos, void *vz,
                                      LongType *zShapeInfo) {
  concatKernelHStack<T>(numArrays, data, inputShapeInfos, vz, zShapeInfo);
}

template <typename T>
SD_HOST void concatKernelHStackGeneric(dim3 &launchDims, cudaStream_t *stream,
                                       int numArrays,
                                       Pointer *data,
                                       Pointer *inputShapeInfos,
                                       void *vz,
                                       LongType *zShapeInfo) {
  execConcatKernelHStack<T>
  <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      numArrays, data, inputShapeInfos, vz, zShapeInfo);
  DebugHelper::checkErrorCode(stream, "concatHStack(...) failed");
}

BUILD_SINGLE_TEMPLATE(
 void concatKernelHStackGeneric,
(dim3 &launchDims, cudaStream_t *stream, int numArrays, sd::Pointer *data,
sd::Pointer *inputShapeInfos, void *vz, sd::LongType *zShapeInfo),
SD_COMMON_TYPES);

}  // namespace sd
