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
static SD_KERNEL void col2imCuda(const void* columns, const sd::LongType* colShapeInfo, void* image,
                                 const sd::LongType* imShapeInfo, const LongType sH, const LongType sW, const LongType pH,
                                 const LongType pW, const LongType dH, const LongType dW) {
  const T* col = reinterpret_cast<const T*>(columns);
  T* im = reinterpret_cast<T*>(image);

  __shared__ sd::LongType kH, kW, oH, oW, *sharedMem;
  __shared__ sd::LongType imLen;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<sd::LongType*>(shmem);

    kH = dH * (colShapeInfo[3] - 1) + 1;
    kW = dW * (colShapeInfo[4] - 1) + 1;

    oH = colShapeInfo[5];
    oW = colShapeInfo[6];

    imLen = shape::length(imShapeInfo);
  }
  __syncthreads();

  auto coords = sharedMem + threadIdx.x * 6;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (sd::LongType i = tid; i < imLen; i += gridDim.x * blockDim.x) {
    shape::index2coords(i, imShapeInfo, coords);

    const auto imOffset = shape::getOffset(imShapeInfo, coords);

    const auto bSiCoffset = coords[0] * colShapeInfo[7] + coords[1] * colShapeInfo[8];

    const sd::LongType imH = coords[2] + pH;
    const sd::LongType imW = coords[3] + pW;

    const sd::LongType colHstart = (imH < kH) ? 0 : (imH - kH) / sH + 1;
    const sd::LongType colWstart = (imW < kW) ? 0 : (imW - kW) / sW + 1;

    const sd::LongType colHend = sd::math::sd_min<sd::LongType>(imH / sH + 1, oH);
    const sd::LongType colWend = sd::math::sd_min<sd::LongType>(imW / sW + 1, oW);

    T val = 0;

    for (coords[4] = colHstart; coords[4] < colHend; ++coords[4]) {
      coords[2] = imH - coords[4] * sH;
      if (coords[2] % dH != 0) continue;

      for (coords[5] = colWstart; coords[5] < colWend; ++coords[5]) {
        coords[3] = imW - coords[5] * sW;
        if (coords[3] % dW != 0) continue;

        val += col[bSiCoffset + (coords[2] / dH) * colShapeInfo[9] + (coords[3] / dW) * colShapeInfo[10] +
                   coords[4] * colShapeInfo[11] + coords[5] * colShapeInfo[12]];
      }
    }
    im[imOffset] = val;
  }
}

////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void col2imCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                               const cudaStream_t* stream, const void* columns, const sd::LongType* colShapeInfo,
                               void* image, const sd::LongType* imShapeInfo, const LongType sH, const LongType sW, const LongType pH,
                               const LongType pW, const LongType dH, const LongType dW) {
  col2imCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(columns, colShapeInfo, image, imShapeInfo, sH,
                                                                        sW, pH, pW, dH, dW);
  sd::DebugHelper::checkGlobalErrorCode( "col2im(...) failed");

}

//////////////////////////////////////////////////////////////////////////
void col2im(sd::LaunchContext& context, const NDArray& col, NDArray& im, const LongType sH, const LongType sW, const LongType pH,
            const LongType pW, const LongType iH, const LongType iW, const LongType dH, const LongType dW) {
  PointersManager manager(&context, "col2im");
  dim3 dims = getCol2imLaunchParams(im,col);

  NDArray::prepareSpecialUse({&im}, {&col});
  BUILD_SINGLE_SELECTOR(im.dataType(), col2imCudaLauncher,
                        (dims.x, dims.y, dims.z, context.getCudaStream(), col.specialBuffer(),
                         col.specialShapeInfo(), im.specialBuffer(), im.specialShapeInfo(), sH, sW, pH, pW, dH, dW),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&im}, {&col});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
