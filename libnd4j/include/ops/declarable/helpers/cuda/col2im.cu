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
// @author raver119@gmail.com, created on 30.11.17.
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/col2im.h>

#include <execution/cuda/LaunchDims.h>


namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// columns [bS, iC, kH, kW, oH, oW] to be de-convoluted to image [bS, iC, iH, iW]
template <typename T>
static SD_KERNEL void col2imCuda(const void* columns, const LongType* colShapeInfo, void* image,
                                 const LongType* imShapeInfo, const LongType sH, const LongType sW, const LongType pH,
                                 const LongType pW, const LongType dH, const LongType dW) {
  const T* col = reinterpret_cast<const T*>(columns);
  T* im = reinterpret_cast<T*>(image);

  __shared__ LongType kH, kW, oH, oW;
  __shared__ LongType imLen;
  __shared__ LongType imRank;
  __shared__ LongType colRank;
  __shared__ LongType* imShape;
  __shared__ LongType* colShape;
  __shared__ LongType* imStride;
  __shared__ LongType* colStride;

  if (threadIdx.x == 0) {
    kH = dH * (colShapeInfo[3] - 1) + 1;
    kW = dW * (colShapeInfo[4] - 1) + 1;
    oH = colShapeInfo[5];
    oW = colShapeInfo[6];
    imLen = shape::length(imShapeInfo);

    // Cache shape information
    imRank = shape::rank(imShapeInfo);
    colRank = shape::rank(colShapeInfo);
    imShape = shape::shapeOf(imShapeInfo);
    colShape = shape::shapeOf(colShapeInfo);
    imStride = shape::stride(imShapeInfo);
    colStride = shape::stride(colShapeInfo);
  }
  __syncthreads();

  LongType coords[SD_MAX_RANK];

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < imLen; i += gridDim.x * blockDim.x) {
    INDEX2COORDS(i, imRank, imShape, coords);

    LongType imOffset;
    COORDS2INDEX(imRank, imStride, coords, imOffset);

    const auto bSiCoffset = coords[0] * colShape[7] + coords[1] * colShape[8];

    const LongType imH = coords[2] + pH;
    const LongType imW = coords[3] + pW;

    const LongType colHstart = (imH < kH) ? 0 : (imH - kH) / sH + 1;
    const LongType colWstart = (imW < kW) ? 0 : (imW - kW) / sW + 1;

    const LongType colHend = sd::math::sd_min<LongType>(imH / sH + 1, oH);
    const LongType colWend = sd::math::sd_min<LongType>(imW / sW + 1, oW);

    T val = static_cast<T>(0);

    for (coords[4] = colHstart; coords[4] < colHend; ++coords[4]) {
      coords[2] = imH - coords[4] * sH;
      if (coords[2] % dH != 0) continue;

      for (coords[5] = colWstart; coords[5] < colWend; ++coords[5]) {
        coords[3] = imW - coords[5] * sW;
        if (coords[3] % dW != 0) continue;

        LongType colOffset;
        COORDS2INDEX(colRank, colStride, coords, colOffset);

        val += col[bSiCoffset + colOffset];
      }
    }
    im[imOffset] = val;
  }
}
////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void col2imCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                               const cudaStream_t* stream, const void* columns, const LongType* colShapeInfo,
                               void* image, const LongType* imShapeInfo, const LongType sH, const LongType sW, const LongType pH,
                               const LongType pW, const LongType dH, const LongType dW) {
  col2imCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(columns, colShapeInfo, image, imShapeInfo, sH,
                                                                        sW, pH, pW, dH, dW);
  DebugHelper::checkGlobalErrorCode( "col2im(...) failed");

}

//////////////////////////////////////////////////////////////////////////
void col2im(LaunchContext& context,  NDArray* input, NDArray* output, const LongType sH, const LongType sW, const LongType pH,
            const LongType pW, const LongType iH, const LongType iW, const LongType dH, const LongType dW){
  PointersManager manager(&context, "col2im");
  dim3 dims = getCol2imLaunchParams(*input,*output);

  NDArray::prepareSpecialUse({input}, {output});
  BUILD_SINGLE_SELECTOR(input->dataType(), col2imCudaLauncher,
                        (dims.x, dims.y, dims.z, context.getCudaStream(), output->specialBuffer(),
                         output->specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), sH, sW, pH, pW, dH, dW),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({input}, {output});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
