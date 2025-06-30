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
#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>

#include "helpers/DebugHelper.h"
#include <system/selective_rendering.h>

namespace sd {
namespace ops {
namespace helpers {
// -------------------------------------------------------------------------------------------------------------- //
// Segment ops linear kernels
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentSumLinearKernel(const void* input, const LongType* inputShape, LongType* starts,
                                            LongType* lengths, LongType numOfClasses, void* output,
                                            const LongType* outputShape) {
 __shared__ T* val;
 __shared__ LongType xLen, zLen, segment, zIndex;
 __shared__ const T* x;
 __shared__ T* z;
 __shared__ int threadsPerSegment, start, finish;

 if (threadIdx.x == 0) {
   threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
   segment = blockIdx.x / threadsPerSegment;
   x = reinterpret_cast<const T*>(input);
   z = reinterpret_cast<T*>(output);

   xLen = shape::length(inputShape);
   zLen = shape::length(outputShape);

   if (segment < numOfClasses) {
     LongType zCoords[SD_MAX_RANK];
     INDEX2COORDS(segment, shape::rank(outputShape), shape::shapeOf(outputShape), zCoords);
     COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zCoords, zIndex);
     if(zIndex >= zLen)
       return;
     start = starts[segment];
     finish = start + lengths[segment];
     LongType xCoords[SD_MAX_RANK];
     INDEX2COORDS(start, shape::rank(inputShape), shape::shapeOf(inputShape), xCoords);
     LongType xOffset;
     COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), xCoords, xOffset);
     z[zIndex] = x[xOffset];
   }
 }
 __syncthreads();

 for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
   LongType xCoords[SD_MAX_RANK];
   INDEX2COORDS(e, shape::rank(inputShape), shape::shapeOf(inputShape), xCoords);
   LongType xOffset;
   COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), xCoords, xOffset);
   if (xOffset >= xLen) return;
   math::atomics::sd_atomicAdd(&z[zIndex], x[xOffset]);
 }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void unsortedSegmentSumLinearKernel(const void* input, const LongType* inputShape,
                                                    const void* indices, const LongType* indicesShape, LongType* starts, LongType* lengths,
                                                    LongType numOfClasses, void* output,
                                                    const LongType* outputShape) {
 __shared__ T* val;
 __shared__ LongType xLen, zLen, segment, zIndex;
 __shared__ const T* x;
 __shared__ T* z;
 __shared__ const I* y;

 if (threadIdx.x == 0) {
   segment = blockIdx.x;
   x = reinterpret_cast<const T*>(input);
   z = reinterpret_cast<T*>(output);
   y = reinterpret_cast<const I*>(indices);
   xLen = shape::length(inputShape);
   zLen = shape::length(outputShape);

   LongType zCoords[SD_MAX_RANK];
   INDEX2COORDS(segment, shape::rank(outputShape), shape::shapeOf(outputShape), zCoords);
   COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zCoords, zIndex);
   if (lengths[segment] > 0) {
     LongType xCoords[SD_MAX_RANK];
     LongType xOffset;
     INDEX2COORDS(starts[segment], shape::rank(inputShape), shape::shapeOf(inputShape), xCoords);
     COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), xCoords, xOffset);
     z[zIndex] = x[xOffset];
   } else {
     z[zIndex] = 0;
   }
 }
 __syncthreads();

 if (lengths[segment] > 0) {
   for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
     LongType xCoords[SD_MAX_RANK];
     LongType yCoords[SD_MAX_RANK];
     LongType xIndex;
     LongType yIndex;

     INDEX2COORDS(e, shape::rank(inputShape), shape::shapeOf(inputShape), xCoords);
     COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), xCoords, xIndex);
     INDEX2COORDS(e, shape::rank(indicesShape), shape::shapeOf(indicesShape), yCoords);
     COORDS2INDEX(shape::rank(indicesShape), shape::stride(indicesShape), yCoords, yIndex);

     if (y[yIndex] == segment && e != starts[segment]) {
       math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex]);
     }
   }
 }
}
// -------------------------------------------------------------------------------------------------------------- //
// SegmentSum kernel
template <typename T, typename I>
static SD_KERNEL void segmentSumTadKernel(void* inputBuf, const LongType* inputShape,
                                         const LongType* inputTads, const LongType* inputTadOffsets,
                                         const I* indices, LongType* starts,
                                         LongType* lengths, LongType numOfClasses, void* outputBuf, const LongType* outputShape,
                                         const LongType* outputTads, const LongType* outputTadOffsets, LongType numIndices) {

 __shared__ LongType len, total;

 if (threadIdx.x == 0) {
   total = shape::sizeAt(inputShape, 0);
   len = shape::length(inputTads);
 }
 __syncthreads();

 for (auto idx = blockIdx.x; idx < total; idx += gridDim.x) {
   auto x = reinterpret_cast<T*>(inputBuf) + inputTadOffsets[idx];
   auto segment = indices[idx];
   auto z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
   auto start = starts[segment];
   auto finish = start + lengths[segment];
   if (lengths[segment] == 0) continue;
   for (auto e = threadIdx.x; e < len; e += blockDim.x) {
     LongType xCoords[SD_MAX_RANK];
     LongType zCoords[SD_MAX_RANK];
     LongType xIndex;
     LongType zIndex;

     INDEX2COORDS(e, shape::rank(inputTads), shape::shapeOf(inputTads), xCoords);
     COORDS2INDEX(shape::rank(inputTads), shape::stride(inputTads), xCoords, xIndex);
     INDEX2COORDS(e, shape::rank(outputTads), shape::shapeOf(outputTads), zCoords);
     COORDS2INDEX(shape::rank(outputTads), shape::stride(outputTads), zCoords, zIndex);

     math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex]);
   }
 }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void segmentSumFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
 auto stream = context->getCudaStream();
 LongType numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
 NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
 NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);
 sd::LongType zero = 0;
 sd::LongType  one = 1;
 sd::LongType  len = indices->lengthOf();
 classesRangesBegs.assign(len);
 classesRangesLens.assign(zero);

 fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

 if (input->isVector() || input->isScalar()) {
   segmentSumLinearKernel<T, I><<<numClasses, input->lengthOf(), numClasses * 32 + 32, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(),
       output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentSumLinearKernel failed");

 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   dim3 segmentTadDims = segmentTad(input->sizeAt(0));
   segmentSumTadKernel<T, I><<<segmentTadDims.y,segmentTadDims.x,segmentTadDims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
       reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(),
       output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentSumTadKernel failed");

   delete dimensions;
 }
}
// -------------------------------------------------------------------------------------------------------------- //
void segmentSumFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices});
 output->nullify();
 auto indicesDType = indices->dataType();
 auto outputDType = input->dataType();
#if SD_IS_PAIR_TYPE_COMPILED(outputDType,indicesDType)
 BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentSumFunctor_, (context, input, indices, output),
                       SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
#endif
 NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static void unsortedSegmentSumFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
 auto stream = context->getCudaStream();
 NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
 NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
 sd::LongType zero = 0;
 sd::LongType  one = 1;
 sd::LongType  len = indices->lengthOf();
 classesRangesBegs.assign(len);
 classesRangesLens.assign(zero);
 dim3 dims = getSegmentSumDims(numOfClasses,indices->lengthOf());
 fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

 if (input->isVector() || input->isScalar()) {
   unsortedSegmentSumLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentSumLinearKernel failed");

 } else {
   output->assign(zero);
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   dim3 dims = segmentTad(input->sizeAt(0));
   segmentSumTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
       reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
       output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentSumTadKernel failed");

   delete dimensions;
   dimensions = nullptr;
 }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentSumFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                              NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices});
 output->nullify();
 auto indicesDType = indices->dataType();
 auto outputDType = input ->dataType();
#if SD_IS_PAIR_TYPE_COMPILED(outputDType,indicesDType)
 BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentSumFunctor_,
                       (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
#endif
 NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
// Backpropagate ops
// -------------------------------------------------------------------------------------------------------------- //
// Sorted sum backpropagate
template <typename T, typename I>
static SD_KERNEL void segmentSumBPLinearKernel(const void* inputBuf, const LongType* inputShape, const void* eps,
                                               const LongType* epsShape, const void* indicesBuf,
                                               const LongType* indicesShape, void* outputBuf,
                                               const LongType* outputShape) {
 __shared__ LongType xLen, gradLen;
 __shared__ sd::LongType inputRank, outputRank, indicesRank, epsRank;
 __shared__ const sd::LongType* inputShapePtr;
 __shared__ const sd::LongType* outputShapePtr;
 __shared__ const sd::LongType* indicesShapePtr;
 __shared__ const sd::LongType* epsShapePtr;
 __shared__ const sd::LongType* inputStridePtr;
 __shared__ const sd::LongType* outputStridePtr;
 __shared__ const sd::LongType* indicesStridePtr;
 __shared__ const sd::LongType* epsStridePtr;

 auto x = reinterpret_cast<const T*>(inputBuf);
 auto y = reinterpret_cast<const I*>(indicesBuf);
 auto z = reinterpret_cast<T*>(outputBuf);
 auto gradOut = reinterpret_cast<const T*>(eps);

 if (threadIdx.x == 0) {
   xLen = shape::length(inputShape);
   gradLen = shape::length(epsShape);

   inputRank = shape::rank(inputShape);
   outputRank = shape::rank(outputShape);
   indicesRank = shape::rank(indicesShape);
   epsRank = shape::rank(epsShape);

   inputShapePtr = shape::shapeOf(inputShape);
   outputShapePtr = shape::shapeOf(outputShape);
   indicesShapePtr = shape::shapeOf(indicesShape);
   epsShapePtr = shape::shapeOf(epsShape);

   inputStridePtr = shape::stride(inputShape);
   outputStridePtr = shape::stride(outputShape);
   indicesStridePtr = shape::stride(indicesShape);
   epsStridePtr = shape::stride(epsShape);
 }
 __syncthreads();

 auto start = blockIdx.x * blockDim.x + threadIdx.x;
 auto step = gridDim.x * blockDim.x;

 for (auto e = start; e < xLen; e += step) {
   LongType zCoords[SD_MAX_RANK];
   LongType xCoords[SD_MAX_RANK];
   LongType yCoords[SD_MAX_RANK];
   LongType zOffset;
   LongType xOffset;
   LongType yOffset;
   LongType gradOffsetO;

   INDEX2COORDS(e, outputRank, outputShapePtr, zCoords);
   COORDS2INDEX(outputRank, outputStridePtr, zCoords, zOffset);
   INDEX2COORDS(e, inputRank, inputShapePtr, xCoords);
   COORDS2INDEX(inputRank, inputStridePtr, xCoords, xOffset);
   INDEX2COORDS(e, indicesRank, indicesShapePtr, yCoords);
   COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yOffset);
   auto classIndex = y[yOffset];
   INDEX2COORDS(classIndex, epsRank, epsShapePtr, zCoords);
   COORDS2INDEX(epsRank, epsStridePtr, zCoords, gradOffsetO);

   z[zOffset] = gradOut[gradOffsetO];
 }
}

template <typename T, typename I>
static SD_KERNEL void segmentSumBPTadKernel(const void* inputBuf, const LongType* inputShape, const void* eps,
                                            const LongType* epsShape, const void* indicesBuf,
                                            const LongType* indicesShape, void* outputBuf,
                                            const LongType* outputShape, const LongType* inputTad,
                                            const LongType* inputOffsets, const LongType* gradOutTad,
                                            const LongType* gradOutOffsets, const LongType* outTad,
                                            const LongType* outOffsets) {
 __shared__ const T* x;
 __shared__ const T* gradOut;
 __shared__ const I* y;
 __shared__ T* z;
 __shared__ LongType xLen, yLen, gradLen, currentLen;
 __shared__ sd::LongType indicesRank;
 __shared__ const sd::LongType* indicesShapePtr;
 __shared__ const sd::LongType* indicesStridePtr;

 if (threadIdx.x == 0) {
   xLen = shape::length(inputShape);
   x = reinterpret_cast<const T*>(inputBuf);
   y = reinterpret_cast<const I*>(indicesBuf);
   z = reinterpret_cast<T*>(outputBuf);
   yLen = shape::length(indicesShape);
   gradOut = reinterpret_cast<const T*>(eps);
   gradLen = shape::length(epsShape);
   currentLen = shape::length(outTad);

   indicesRank = shape::rank(indicesShape);
   indicesShapePtr = shape::shapeOf(indicesShape);
   indicesStridePtr = shape::stride(indicesShape);
 }
 __syncthreads();

 for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
   LongType yCoords[SD_MAX_RANK];
   LongType yIndex;
   INDEX2COORDS(i, indicesRank, indicesShapePtr, yCoords);
   COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yIndex);
   auto segment = y[yIndex];
   auto currentOut = z + outOffsets[i];
   auto outGrad = gradOut + gradOutOffsets[segment];

   for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
     currentOut[e] = outGrad[e];
   }
 }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
Status segmentSumFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                           NDArray* output) {
 auto stream = context->getCudaStream();
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 if (input->isVector()  || input->isScalar()) {
   LongType loop_size = input->lengthOf();
   auto numOfClasses = gradOut->lengthOf();
   segmentSumBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentSumBPLinearKernel failed");

 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   auto gradOutTads = packGradOut->specialShapeInfo();
   auto gradOutTadOffsets = packGradOut->specialOffsets();

   segmentSumBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
       inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
   sd::DebugHelper::checkErrorCode(stream, "segmentSumBPTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
 return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //

Status segmentSumFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                          NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 auto indicesDType = indices->dataType();
 auto outputDType = output->dataType();
#if SD_IS_PAIR_TYPE_COMPILED(indicesDType,outputDType)
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentSumFunctorBP_,
                       (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
#endif
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

template <typename T, typename I>
static Status unsortedSegmentSumFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                          NDArray* gradOut,
                                          LongType numOfClasses, NDArray* output) {
 auto stream = context->getCudaStream();
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 if (input->isVector()  || input->isScalar()) {
   LongType loop_size = input->lengthOf();
   auto numOfClasses = gradOut->lengthOf();
   segmentSumBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentSumBPLinearKernel failed");

 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   auto gradOutTads = packGradOut->specialShapeInfo();
   auto gradOutTadOffsets = packGradOut->specialOffsets();

   segmentSumBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
       inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
   sd::DebugHelper::checkErrorCode(stream, "segmentSumBPTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
 return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentSumFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                  LongType numOfClasses, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 auto indicesDType = indices->dataType();
 auto outputDType = output->dataType();
#if SD_IS_PAIR_TYPE_COMPILED(outputDType,indicesDType)
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentSumFunctorBP_,
                       (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
#endif
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});

}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
