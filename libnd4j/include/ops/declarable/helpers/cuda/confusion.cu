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
#include <memory/cuda/CudaMemoryPool.h>

#include <ops/declarable/helpers/confusion.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
SD_KERNEL static void copyBuffers(LongType* destination, void const* source, LongType bufferLength) {
 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
 const auto step = gridDim.x * blockDim.x;
 const T * sourceCast = reinterpret_cast<T const*>(source);
 for (int t = tid; t < bufferLength; t += step) {
   destination[t] = static_cast<LongType>(sourceCast[t]);
 }


}




template <typename T>
SD_KERNEL static void confusionFunctorKernel(LongType* labelsBuffer, LongType* predictionBuffer, LongType bufferLength, void const* weightsBuffer, void* outputBuffer,
                                             const LongType* tadShape, const LongType* tadOffsets) {
 __shared__ int arrIdx, blocksPerArr;
 __shared__ T* z;
 __shared__ T const* w;
 __shared__ LongType *zShapeInfo, *xShapeInfo, arrLen;
 __shared__ LongType tadRank;
 __shared__ LongType* tadShapePtr;
 __shared__ LongType* tadStridePtr;

 if (threadIdx.x == 0) {
   z = reinterpret_cast<T*>(outputBuffer);
   w = reinterpret_cast<T const*>(weightsBuffer);
   arrLen = shape::length(tadShape);

   // Cache shape information
   tadRank = shape::rank(tadShape);
   tadShapePtr = shape::shapeOf(tadShape);
   tadStridePtr = shape::stride(tadShape);
 }
 __syncthreads();

 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
 const auto step = gridDim.x * blockDim.x;
 LongType predCoords[SD_MAX_RANK];
 LongType predOffset;

 for (int t = tid; t < bufferLength; t += step) {
   auto label = labelsBuffer[t];
   auto pred = predictionBuffer[t];
   auto tZ = z + tadOffsets[label];
   T val = (weightsBuffer == nullptr ? (T)1.0f : w[t]);

   INDEX2COORDS(pred, tadRank, tadShapePtr, predCoords);
   COORDS2INDEX(tadRank, tadStridePtr, predCoords, predOffset);
   sd::math::atomics::sd_atomicAdd(&tZ[predOffset], val);
 }
}
template <typename X, typename Z>
void _confusionFunctor(LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights,
                      NDArray* output) {
 auto stream = context->getCudaStream();

 auto pack = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), 1);
 PointersManager manager(context, "helpers::confusion");
predictions->syncToDevice();
  LongType* labelsLongBuffer = labels->dataType() == INT64 ? (LongType*)labels->specialBuffer() : nullptr;
  LongType* predictionLongBuffer =
     predictions->dataType() == INT64 ? (LongType*)predictions->specialBuffer() : nullptr;



 dim3 conf = getLaunchDims("confusion_matrix");
 if (labelsLongBuffer == nullptr) {
   int devId = 0; cudaGetDevice(&devId);
   labelsLongBuffer = reinterpret_cast<LongType*>(sd::memory::CudaMemoryPool::getInstance().allocate(labels->lengthOf() * sizeof(LongType), devId, nullptr));
   if (labelsLongBuffer == nullptr) THROW_EXCEPTION("Cannot allocate memory for labels long buffer");
   // copy with type conversion
   copyBuffers<X><<<conf.x, conf.y, conf.z, *stream>>>(labelsLongBuffer, labels->specialBuffer(), labels->lengthOf());
   sd::DebugHelper::checkGlobalErrorCode("copyBuffers  failed");

 }

 if (predictionLongBuffer == nullptr) {
   int devId2 = 0; cudaGetDevice(&devId2);
   predictionLongBuffer = reinterpret_cast<LongType*>(sd::memory::CudaMemoryPool::getInstance().allocate(predictions->lengthOf() * sizeof(LongType), devId2, nullptr));
   if (predictionLongBuffer == nullptr) THROW_EXCEPTION("Cannot allocate memory for predictions long buffer");
   // copy with type conversion
   copyBuffers<X>
       <<<256, 512, 1024, *stream>>>(predictionLongBuffer, predictions->specialBuffer(), predictions->lengthOf());
   sd::DebugHelper::checkGlobalErrorCode("copyBuffers  failed");

 }

 manager.synchronize();



 auto bufferLength = labels->lengthOf();
 dim3 launchDims = getLaunchDims("confusionMatrix");
 confusionFunctorKernel<Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
     labelsLongBuffer, predictionLongBuffer, bufferLength, weights != nullptr ? weights->specialBuffer() : nullptr,
     output->specialBuffer(), pack->specialShapeInfo(), pack->specialOffsets());
  sd::DebugHelper::checkGlobalErrorCode("confusionFunctorKernel  failed");

 manager.synchronize();

 if (predictionLongBuffer != predictions->specialBuffer()) {
   int devIdFree1 = 0; cudaGetDevice(&devIdFree1);
   sd::memory::CudaMemoryPool::getInstance().free(predictionLongBuffer, devIdFree1, nullptr);
 }

 if (labelsLongBuffer != labels->specialBuffer()) {
   int devIdFree2 = 0; cudaGetDevice(&devIdFree2);
   sd::memory::CudaMemoryPool::getInstance().free(labelsLongBuffer, devIdFree2, nullptr);
 }
}

void confusionFunctor(LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights,
                     NDArray* output) {
 auto xType = predictions->dataType();
 auto zType = output->dataType();  // weights can be null
 NDArray::prepareSpecialUse({output}, {labels, predictions, weights});
 BUILD_DOUBLE_SELECTOR(xType, zType, _confusionFunctor, (context, labels, predictions, weights, output),
                       SD_INDEXING_TYPES, SD_NUMERIC_TYPES);
 NDArray::registerSpecialUse({output}, {labels, predictions, weights});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
