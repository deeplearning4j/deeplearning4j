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
  SD_DEVICE void concatKernelVStack(int numArrays,
                                    Pointer* data,
                                    Pointer* inputShapeInfos,
                                    void* vz,
                                    LongType* zShapeInfo) {
    auto z           = reinterpret_cast<T*>(vz);
    auto inputShapes = reinterpret_cast<LongType**>(inputShapeInfos);
    auto inputData   = reinterpret_cast<T**>(data);

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ sd::LongType zRank;
    __shared__ const sd::LongType* zShapePtr;
    __shared__ const sd::LongType* zStridePtr;

    // We'll store the rank/shape/stride of each input vector once per array
    // to avoid repeated calls inside the loop
    __shared__ sd::LongType inputRank;
    __shared__ const sd::LongType* inputShapePtr;
    __shared__ const sd::LongType* inputStridePtr;
    __shared__ sd::LongType rowLength;  // length of each input vector

    if (threadIdx.x == 0) {
      zRank      = shape::rank(zShapeInfo);
      zShapePtr  = shape::shapeOf(zShapeInfo);
      zStridePtr = shape::stride(zShapeInfo);
    }
    __syncthreads();

    // For each array, we assume it is a vector that will form one row of z
    for (int r = blockIdx.x; r < numArrays; r += gridDim.x) {
      // Single thread loads shape info for the current input array
      if (threadIdx.x == 0) {
        inputRank       = shape::rank(inputShapes[r]);
        inputShapePtr   = shape::shapeOf(inputShapes[r]);
        inputStridePtr  = shape::stride(inputShapes[r]);
        rowLength       = shape::length(inputShapes[r]);
      }
      __syncthreads();

      // Each thread copies part of the row for this array
      for (sd::LongType i = tid; i < rowLength; i += blockDim.x * gridDim.x) {
        // We'll do coordinate transforms to find the correct offsets:

        // 1) Input offset
        sd::LongType inCoords[SD_MAX_RANK];
        INDEX2COORDS(i, inputRank, inputShapePtr, inCoords);

        sd::LongType inOffset;
        COORDS2INDEX(inputRank, inputStridePtr, inCoords, inOffset);

        // 2) Output offset
        // The "row" dimension is r, the "column" dimension is i.
        sd::LongType outCoords[SD_MAX_RANK];
        outCoords[0] = r;    // row
        outCoords[1] = i;    // column

        sd::LongType outOffset;
        COORDS2INDEX(zRank, zStridePtr, outCoords, outOffset);

        z[outOffset] = inputData[r][inOffset];
      }
      __syncthreads();
    }
  }

  template <typename T>
  SD_KERNEL void execConcatKernelVStack(
      int numArrays,
      Pointer* data,
      Pointer* inputShapeInfos,
      void* vz,
      LongType* zShapeInfo) {

    concatKernelVStack<T>(
        numArrays,
        data,
        inputShapeInfos,
        vz,
        zShapeInfo);
  }

  template <typename T>
  SD_HOST void concatKernelVStackGeneric(
      dim3 &launchDims,
      cudaStream_t *stream,
      int numArrays,
      Pointer* data,
      Pointer* inputShapeInfos,
      void* vz,
      LongType* zShapeInfo) {

    execConcatKernelVStack<T>
        <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            numArrays,
            data,
            inputShapeInfos,
            vz,
            zShapeInfo);

    DebugHelper::checkErrorCode(stream, "concatVStack(...) failed");
  }

  BUILD_SINGLE_TEMPLATE(
       void concatKernelVStackGeneric,
      (dim3 & launchDims, cudaStream_t *stream, int numArrays, sd::Pointer *data,
       sd::Pointer *inputShapeInfos, void *vz, sd::LongType *zShapeInfo),
      SD_COMMON_TYPES);

}  // namespace sd
