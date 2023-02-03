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
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/confusion.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
SD_KERNEL static void copyBuffers(sd::LongType* destination, void const* source, sd::LongType bufferLength) {
 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
 const auto step = gridDim.x * blockDim.x;
 const T * sourceCast = reinterpret_cast<T const*>(source);
 for (int t = tid; t < bufferLength; t += step) {
   destination[t] = static_cast<sd::LongType>(sourceCast[t]);
 }


}


SD_KERNEL static void printBuffers(sd::LongType* buffer, sd::LongType bufferLength) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;
  for (int t = tid; t < bufferLength; t += step) {
  }


}

template <typename T>
SD_KERNEL static void confusionFunctorKernel(sd::LongType* labelsBuffer, sd::LongType* predictionBuffer,
                                            sd::LongType bufferLength, void const* weightsBuffer, void* outputBuffer,
                                            const sd::LongType* tadShape, const sd::LongType* tadOffsets) {
 __shared__ int arrIdx, blocksPerArr;
 __shared__ T* z;
 __shared__ T const* w;
 __shared__ sd::LongType *zShapeInfo, *xShapeInfo, arrLen;

 if (threadIdx.x == 0) {
   z = reinterpret_cast<T*>(outputBuffer);
   w = reinterpret_cast<T const*>(weightsBuffer);
   arrLen = shape::length(tadShape);
 }
 __syncthreads();

 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
 const auto step = gridDim.x * blockDim.x;
 for (int t = tid; t < bufferLength; t += step) {
   auto label = labelsBuffer[t];     //->e<sd::LongType>(j);
   auto pred = predictionBuffer[t];  //->e<sd::LongType>(j);
   auto tZ = z + tadOffsets[label];
   T var = (weightsBuffer == nullptr ? (T)1.0f : w[t]);

   auto idx = shape::getIndexOffset(pred, tadShape);
   tZ[idx] = val;
 }
}

template <typename X, typename Z>
void _confusionFunctor(sd::LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights,
                      NDArray* output) {
 auto stream = context->getCudaStream();

 auto pack = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), 1);
 PointersManager manager(context, "helpers::confusion");
predictions->syncToDevice();
 sd::LongType* labelsLongBuffer =
     labels->dataType() == sd::DataType::INT64 ? (sd::LongType*)labels->specialBuffer() : nullptr;
 sd::LongType* predictionLongBuffer =
     predictions->dataType() == sd::DataType::INT64 ? (sd::LongType*)predictions->specialBuffer() : nullptr;



 if (labelsLongBuffer == nullptr) {
   auto err = cudaMalloc(&labelsLongBuffer, labels->lengthOf() * sizeof(sd::LongType));
   if (err != 0) throw sd::cuda_exception::build("Cannot allocate memory for labels long buffer", err);
   // copy with type conversion
   copyBuffers<X><<<256, 512, 1024, *stream>>>(labelsLongBuffer, labels->specialBuffer(), labels->lengthOf());
 }

 if (predictionLongBuffer == nullptr) {
   auto err = cudaMalloc(&predictionLongBuffer, predictions->lengthOf() * sizeof(sd::LongType));
   if (err != 0) throw sd::cuda_exception::build("Cannot allocate memory for predictions long buffer", err);
   // copy with type conversion
   copyBuffers<X>
       <<<256, 512, 1024, *stream>>>(predictionLongBuffer, predictions->specialBuffer(), predictions->lengthOf());
 }

 manager.synchronize();



 auto bufferLength = labels->lengthOf();
 dim3 launchDims(32, 32, 1024);
 confusionFunctorKernel<Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
     labelsLongBuffer, predictionLongBuffer, bufferLength, weights != nullptr ? weights->specialBuffer() : nullptr,
     output->specialBuffer(), pack.specialShapeInfo(), pack.specialOffsets());

 manager.synchronize();

 if (predictionLongBuffer != predictions->specialBuffer()) {
   cudaError_t err = cudaFree(predictionLongBuffer);
   if (err != 0) throw sd::cuda_exception::build("Cannot deallocate memory for predictions long buffer", err);
 }

 if (labelsLongBuffer != labels->specialBuffer()) {
   cudaError_t err = cudaFree(labelsLongBuffer);
   if (err != 0) throw sd::cuda_exception::build("Cannot deallocate memory for labels long buffer", err);
 }
}

void confusionFunctor(sd::LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights,
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
