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
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>

#include <ops/declarable/helpers/confusion.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {

// labels and predictions arrive as contiguous INT64 vectors and the weights, when present,
// as a contiguous vector of the output type: the caller converts them (see _confusionFunctor).
template <typename Z>
SD_KERNEL static void confusionFunctorKernel(const LongType* labelsBuffer, const LongType* predictionBuffer,
                                             LongType bufferLength, const Z* weightsBuffer, Z* outputBuffer,
                                             const LongType* tadShape, const LongType* tadOffsets) {
 __shared__ LongType tadRank;
 __shared__ LongType* tadShapePtr;
 __shared__ LongType* tadStridePtr;

 if (threadIdx.x == 0) {
   tadRank = shape::rank(tadShape);
   tadShapePtr = shape::shapeOf(tadShape);
   tadStridePtr = shape::stride(tadShape);
 }
 __syncthreads();

 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
 const auto step = gridDim.x * blockDim.x;
 LongType predCoords[SD_MAX_RANK];
 LongType predOffset;

 for (LongType t = tid; t < bufferLength; t += step) {
   auto label = labelsBuffer[t];
   auto pred = predictionBuffer[t];
   auto tZ = outputBuffer + tadOffsets[label];
   Z val = (weightsBuffer == nullptr ? static_cast<Z>(1) : weightsBuffer[t]);

   INDEX2COORDS(pred, tadRank, tadShapePtr, predCoords);
   COORDS2INDEX(tadRank, tadStridePtr, predCoords, predOffset);
   sd::math::atomics::sd_atomicAdd(&tZ[predOffset], val);
 }
}

template <typename Z>
static void _confusionFunctor(LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights,
                              NDArray* output) {
 auto stream = context->getCudaStream();
 auto pack = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), 1);
 PointersManager manager(context, "helpers::confusion");

 // Contiguous copies in the types the kernel reads: whatever the inputs' own types and
 // strides, labels and predictions become INT64 indices and weights take the output's type.
 NDArray* labelsLong = labels->cast(INT64);
 NDArray* predictionsLong = predictions->cast(INT64);
 NDArray* weightsZ = weights != nullptr ? weights->cast(output->dataType()) : nullptr;

 NDArray::prepareSpecialUse({output}, {labelsLong, predictionsLong, weightsZ});
 dim3 launchDims = getLaunchDims("confusionMatrix");
 confusionFunctorKernel<Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
     reinterpret_cast<const LongType*>(labelsLong->specialBuffer()),
     reinterpret_cast<const LongType*>(predictionsLong->specialBuffer()), labels->lengthOf(),
     weightsZ != nullptr ? reinterpret_cast<const Z*>(weightsZ->specialBuffer()) : nullptr,
     reinterpret_cast<Z*>(output->specialBuffer()), pack->specialShapeInfo(), pack->specialOffsets());
 sd::DebugHelper::checkGlobalErrorCode("confusionFunctorKernel  failed");
 NDArray::registerSpecialUse({output}, {labelsLong, predictionsLong, weightsZ});

 // The kernel reads the copies asynchronously: finish it before they are freed.
 manager.synchronize();
 delete labelsLong;
 delete predictionsLong;
 delete weightsZ;
}

void confusionFunctor(LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights,
                     NDArray* output) {
 auto zType = output->dataType();
 NDArray::prepareSpecialUse({output}, {labels, predictions, weights});
 BUILD_SINGLE_SELECTOR(zType, _confusionFunctor, (context, labels, predictions, weights, output), SD_NUMERIC_TYPES);
 NDArray::registerSpecialUse({output}, {labels, predictions, weights});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
