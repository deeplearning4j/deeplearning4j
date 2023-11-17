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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/adjust_hue.h>
#include <ops/declarable/helpers/adjust_saturation.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
static void SD_KERNEL adjustSaturationCuda(const void* vx, const LongType* xShapeInfo,
                                           const LongType* xTadOffsets, void* vz, const LongType* zShapeInfo,
                                           const LongType* zTadOffsets, const LongType numOfTads,
                                           const T factor, const LongType dimC) {
  const T* x = reinterpret_cast<const T*>(vx);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ LongType rank;
  __shared__ LongType xDimCstride, zDimCstride;

  if (threadIdx.x == 0) {
    rank = shape::rank(xShapeInfo);
    xDimCstride = shape::stride(xShapeInfo)[dimC];
    zDimCstride = shape::stride(zShapeInfo)[dimC];
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < numOfTads; i += gridDim.x * blockDim.x) {
    const T* xTad = x + xTadOffsets[i];
    T* zTad = z + zTadOffsets[i];

    T h, s, v;

    rgbToHsv<T>(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], h, s, v);

    s *= factor;
    if (s > 1.f)
      s = 1.f;
    else if (s < 0.f)
      s = 0.f;

    hsvToRgb<T>(h, s, v, zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static SD_HOST void adjustSaturationCudaLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                                 const cudaStream_t* stream, const void* vx,
                                                 const LongType* xShapeInfo, const LongType* xTadOffsets,
                                                 void* vz, const LongType* zShapeInfo,
                                                 const LongType* zTadOffsets, const LongType numOfTads,
                                                 const NDArray* factorScalarArr, const LongType dimC) {
  adjustSaturationCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(
      vx, xShapeInfo, xTadOffsets, vz, zShapeInfo, zTadOffsets, numOfTads, factorScalarArr->e<T>(0), dimC);
  sd::DebugHelper::checkGlobalErrorCode("adjustSaturation  failed");

}

////////////////////////////////////////////////////////////////////////
void adjustSaturation(LaunchContext* context, const NDArray* input, const NDArray* factorScalarArr, NDArray* output,
                      const LongType dimC) {
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), {dimC});
  auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), {dimC});

  const LongType numOfTads = packX->numberOfTads();

  dim3 adjustDims = getAdjustDims(numOfTads);

  PointersManager manager(context, "adjustSaturation");

  NDArray::prepareSpecialUse({output}, {input, factorScalarArr});
  BUILD_SINGLE_SELECTOR(input->dataType(), adjustSaturationCudaLauncher,
                        (adjustDims.x,adjustDims.y, context->getCudaStream(), input->specialBuffer(),
                         input->specialShapeInfo(), packX->platformOffsets(), output->specialBuffer(),
                         output->specialShapeInfo(), packZ->platformOffsets(), numOfTads, factorScalarArr, dimC),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({output}, {input, factorScalarArr});

  manager.synchronize();
}



}  // namespace helpers
}  // namespace ops
}  // namespace sd
