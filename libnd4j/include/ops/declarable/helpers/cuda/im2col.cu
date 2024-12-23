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
// Created by raver119 on 30.11.17.
//
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/im2col.h>

#include <execution/cuda/LaunchDims.h>


namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// input [bS, iC, iH, iW] is convoluted to output [bS, iC, kH, kW, oH, oW]
template <typename T>
SD_KERNEL static void im2colCuda(const void *image, void *columns, const LongType *imShapeInfo,
                                 const LongType *colShapeInfo, const LongType sH, const LongType sW, const LongType pH,
                                 const LongType pW, const LongType dH, const LongType dW, const double zeroPadValD) {
  T zeroPadVal = static_cast<T>(zeroPadValD);  // Value to use when value is padding
  const auto im = reinterpret_cast<const T *>(image);
  auto col = reinterpret_cast<T *>(columns);

  // Shared memory caching
  __shared__ LongType colLen, imLen, iH, iW;
  __shared__ LongType imRank, colRank;
  __shared__ const LongType *imShapePtr, *imStridePtr;
  __shared__ const LongType *colShapePtr, *colStridePtr;

  if (threadIdx.x == 0) {
    colRank = 6;
    imRank = 4;

    colLen = shape::length(colShapeInfo);
    imLen = shape::length(imShapeInfo);

    iH = shape::shapeOf(imShapeInfo)[2];
    iW = shape::shapeOf(imShapeInfo)[3];

    imShapePtr = shape::shapeOf(imShapeInfo);
    imStridePtr = shape::stride(imShapeInfo);

    colShapePtr = shape::shapeOf(colShapeInfo);
    colStridePtr = shape::stride(colShapeInfo);
  }
  __syncthreads();

  const auto colInd = threadIdx.x + blockIdx.x * blockDim.x;

  if (colInd >= colLen) return;  // Boundary check for threads

  LongType coords[SD_MAX_RANK];

  // Calculate coordinates and offsets
  INDEX2COORDS(colInd, colRank, colShapePtr, coords);

  LongType colOffset;
  COORDS2INDEX(colRank, colStridePtr, coords, colOffset);

  coords[2] = (-pH + coords[2] * dH) + coords[4] * sH;  // imH
  coords[3] = (-pW + coords[3] * dW) + coords[5] * sW;  // imW

  // Check bounds and assign appropriate values
  if (coords[2] >= iH || coords[3] >= iW || coords[2] < 0 || coords[3] < 0) {
    if (colOffset < colLen)
      col[colOffset] = zeroPadVal;
  } else {
    LongType imOffset;
    COORDS2INDEX(imRank, imStridePtr, coords, imOffset);
    if (imOffset < imLen && colOffset < colLen)
      col[colOffset] = im[imOffset];
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void im2colCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
                               LaunchContext &context, const void *image, void *columns,
                               const LongType *imShapeInfo, const LongType *colShapeInfo, LongType sH,
                               LongType sW, LongType pH, LongType pW, LongType dH, LongType dW, double zeroPadVal) {
  im2colCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory /* rank of columns = 6 */, *context.getCudaStream()>>>(
      image, columns, imShapeInfo, colShapeInfo, sH, sW, pH, pW, dH, dW, zeroPadVal);
  DebugHelper::checkErrorCode(context.getCudaStream(), "im2colCuda(...) failed");

}

//////////////////////////////////////////////////////////////////////////
void im2col(LaunchContext &context, NDArray&image, NDArray &columns, const LongType kH, const LongType kW,
            const LongType sH, const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW,
            NDArray&arrZeroPadVal) {
  PointersManager manager(&context, "im2col");

  dim3 im2colDevs = getim2ColLaunchParams(columns);
  NDArray::prepareSpecialUse({&columns}, {&image});
  BUILD_SINGLE_SELECTOR(
      columns.dataType(), im2colCudaLauncher,
      (im2colDevs.x, im2colDevs.y,im2colDevs.z, context, image.specialBuffer(), columns.specialBuffer(),
          image.specialShapeInfo(), columns.specialShapeInfo(), sH, sW, pH, pW, dH, dW, arrZeroPadVal.e<double>(0)),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&columns}, {&image});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
