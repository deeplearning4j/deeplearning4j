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
  // image [bS, iC, iH, iW] is convoluted to columns [bS, iC, kH, kW, oH, oW]
  const T zeroPadVal = static_cast<T>(zeroPadValD);
  const auto im = reinterpret_cast<const T *>(image);
  auto col = reinterpret_cast<T *>(columns);

  constexpr int colRank = 6;
  constexpr int imRank = 4;
  const LongType colLen = shape::length(colShapeInfo);
  const LongType iH = shape::shapeOf(imShapeInfo)[2];
  const LongType iW = shape::shapeOf(imShapeInfo)[3];
  const LongType *colShape = shape::shapeOf(colShapeInfo);
  const LongType *colStride = shape::stride(colShapeInfo);
  const LongType *imStride = shape::stride(imShapeInfo);

  LongType coords[colRank];
  for (LongType colInd = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; colInd < colLen;
       colInd += static_cast<LongType>(gridDim.x) * blockDim.x) {
    INDEX2COORDS(colInd, colRank, colShape, coords);

    // Offsets come from in-range coordinates and each array's own strides, so they address the
    // array's elements whatever its layout: a view's offsets may exceed its length.
    LongType colOffset;
    COORDS2INDEX(colRank, colStride, coords, colOffset);

    coords[2] = (-pH + coords[2] * dH) + coords[4] * sH;  // imH
    coords[3] = (-pW + coords[3] * dW) + coords[5] * sW;  // imW

    if (coords[2] >= iH || coords[3] >= iW || coords[2] < 0 || coords[3] < 0) {
      col[colOffset] = zeroPadVal;
    } else {
      LongType imOffset;
      COORDS2INDEX(imRank, imStride, coords, imOffset);
      col[colOffset] = im[imOffset];
    }
  }
}

template <typename T>
static void im2colCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
                               LaunchContext &context, const void *image, void *columns,
                               const LongType *imShapeInfo, const LongType *colShapeInfo, LongType sH,
                               LongType sW, LongType pH, LongType pW, LongType dH, LongType dW, double zeroPadVal) {
  auto stream = context.getCudaStream();
  im2colCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(
      image, columns, imShapeInfo, colShapeInfo, sH, sW, pH, pW, dH, dW, zeroPadVal);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("im2colCuda(...) failed");
  }
}

void im2col(LaunchContext &context, NDArray&image, NDArray &columns, const LongType kH, const LongType kW,
            const LongType sH, const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW,
            NDArray&arrZeroPadVal) {
  if (columns.lengthOf() == 0) return;
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
