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
//  @author sgazeos@gmail.com
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/helpers/nth_element.h>

#include "array/NDArrayFactory.h"
#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_KERNEL void fillUpElementKernel(void* outputBuffer, LongType const* outputShapeInfo, void* inputBuffer,
                                          LongType const* inputShapeInfo, LongType const* pTadShape,
                                          LongType const* pTadOffsets, LongType n) {
  __shared__ LongType bufferLength;

  auto z = reinterpret_cast<T*>(outputBuffer);
  auto x = reinterpret_cast<T*>(inputBuffer);

  if (threadIdx.x == 0) bufferLength = shape::length(outputShapeInfo);

  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;
  for (int t = tid; t < bufferLength; t += step) {
    auto tX = x + pTadOffsets[t];
    z[shape::getIndexOffset(t, outputShapeInfo)] = tX[shape::getIndexOffset(n, pTadShape)];  // tX];
  }
}

template <typename T>
void nthElementFunctor_(LaunchContext* context, NDArray* input, LongType n, NDArray* output, bool reverse) {
  NDArray::prepareSpecialUse({output}, {input});
  NDArray sortedVals(*input);
  Pointer params[2];
  params[0] = context;
  params[1] = context->getCudaStream();
  // Nth element in sorted sequence : basic algorithm sort and retrieve nth element in sorted
  if (input->isVector()) {
    sort(params, &sortedVals, reverse);

    cudaMemcpy(reinterpret_cast<T*>(output->specialBuffer()), reinterpret_cast<T*>(sortedVals.specialBuffer()) + n,
               sizeof(T), cudaMemcpyDeviceToDevice);
  } else {  // rank greater than 1
    std::vector<LongType> lastDims(
        {input->rankOf() - 1});
    NDArray *dimData = NDArrayFactory::create_<LongType>('c',{2},lastDims, context);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(sortedVals.shapeInfo(), &lastDims);

    auto pTadShape = packX->specialShapeInfo();
    auto pTadShapeH = packX->primaryShapeInfo();
    auto pTadOffsets = packX->specialOffsets();
    sortTad(params, &sortedVals,
            reinterpret_cast<sd::LongType *>(lastDims.data()),
           lastDims.size(),
            const_cast<sd::LongType *>(pTadShape),
            const_cast<sd::LongType *>(pTadOffsets),
            reverse);
    sortedVals.tickWriteDevice();
    sortedVals.syncToHost();
    auto stream = context->getCudaStream();
    dim3 launchDims = getLaunchDims("nth_element_fill");
    fillUpElementKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(output->specialBuffer(), output->specialShapeInfo(),
                                                      sortedVals.specialBuffer(), sortedVals.specialShapeInfo(),
                                                      pTadShape, pTadOffsets, n);
    sd::DebugHelper::checkErrorCode(stream, "fillUpElementKernel failed");

  }
  NDArray::registerSpecialUse({output}, {input});
}
void nthElementFunctor(LaunchContext* context, NDArray* input, LongType n, NDArray* output, bool reverse) {
  BUILD_SINGLE_SELECTOR(input->dataType(), nthElementFunctor_, (context, input, n, output, reverse), SD_COMMON_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
