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
namespace sd {
namespace ops {
namespace helpers {
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void unsortedSegmentSqrtNLinearKernel(T* input, LongType const* inputShape, I* indices,
                                                       LongType const* indicesShape, LongType* starts,
                                                       LongType* lengths, LongType numOfClasses, T* output,
                                                       LongType const* outputShape) {
  __shared__ LongType xLen, zLen;
  __shared__ sd::LongType inputRank, outputRank, indicesRank;
  __shared__ const sd::LongType* inputShapePtr;
  __shared__ const sd::LongType* outputShapePtr;
  __shared__ const sd::LongType* indicesShapePtr;
  __shared__ const sd::LongType* inputStridePtr;
  __shared__ const sd::LongType* outputStridePtr;
  __shared__ const sd::LongType* indicesStridePtr;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    inputRank = shape::rank(inputShape);
    outputRank = shape::rank(outputShape);
    indicesRank = shape::rank(indicesShape);

    inputShapePtr = shape::shapeOf(inputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    indicesShapePtr = shape::shapeOf(indicesShape);

    inputStridePtr = shape::stride(inputShape);
    outputStridePtr = shape::stride(outputShape);
    indicesStridePtr = shape::stride(indicesShape);
  }
  __syncthreads();

  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  LongType yCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];
  LongType xCoords[SD_MAX_RANK];
  LongType yIndex;
  LongType zIndex;
  LongType xIndex;

  for (auto idx = start; idx < xLen; idx += step) {
    INDEX2COORDS(idx, indicesRank, indicesShapePtr, yCoords);
    COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yIndex);
    auto segment = indices[yIndex];
    INDEX2COORDS(segment, outputRank, outputShapePtr, zCoords);
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zIndex);
    if (lengths[segment] == 0) continue;
    INDEX2COORDS(idx, inputRank, inputShapePtr, xCoords);
    COORDS2INDEX(inputRank, inputStridePtr, xCoords, xIndex);
    if (xIndex >= xLen) continue;
    math::atomics::sd_atomicAdd(&output[zIndex], input[xIndex] / math::sd_sqrt<LongType, T>(lengths[segment]));
  }
}

template <typename T, typename I>
static SD_KERNEL void segmentSqrtNTadKernel(T* inputBuf, LongType const* inputShape, LongType const* inputTads,
                                            LongType const* inputTadOffsets, I* indices, LongType* starts,
                                            LongType* lengths, LongType numOfClasses, void* outputBuf,
                                            LongType const* outputShape, LongType const* outputTads,
                                            LongType const* outputTadOffsets, LongType numIndices) {
  if(blockIdx.x >= numIndices)
    return;

  __shared__ LongType len, total;
  __shared__ sd::LongType inputTadRank, outputTadRank;
  __shared__ const sd::LongType* inputTadShapePtr;
  __shared__ const sd::LongType* outputTadShapePtr;
  __shared__ const sd::LongType* inputTadStridePtr;
  __shared__ const sd::LongType* outputTadStridePtr;

  if (threadIdx.x == 0) {
    total = shape::sizeAt(inputShape, 0);
    len = shape::length(inputTads);

    inputTadRank = shape::rank(inputTads);
    outputTadRank = shape::rank(outputTads);

    inputTadShapePtr = shape::shapeOf(inputTads);
    outputTadShapePtr = shape::shapeOf(outputTads);

    inputTadStridePtr = shape::stride(inputTads);
    outputTadStridePtr = shape::stride(outputTads);
  }
  __syncthreads();

  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];
  LongType xIndex;
  LongType zIndex;

  for (auto idx = blockIdx.x; idx < total; idx += gridDim.x) {
    auto segment = indices[idx];
    auto x = inputBuf + inputTadOffsets[idx];
    auto z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
    auto start = starts[segment];
    auto finish = start + lengths[segment];

    for (auto e = threadIdx.x; e < len; e += blockDim.x) {
      INDEX2COORDS(e, inputTadRank, inputTadShapePtr, inputCoords);
      COORDS2INDEX(inputTadRank, inputTadStridePtr, inputCoords, xIndex);
      INDEX2COORDS(e, outputTadRank, outputTadShapePtr, outputCoords);
      COORDS2INDEX(outputTadRank, outputTadStridePtr, outputCoords, zIndex);
      math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex] / math::sd_sqrt<LongType, T>(lengths[segment]));
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static void unsortedSegmentSqrtNFunctor_(LaunchContext* context, NDArray* input, NDArray* indices,
                                        LongType numOfClasses, NDArray* output) {
 auto stream = context->getCudaStream();
 NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
 NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
 sd::LongType zero = 0;
 sd::LongType  one = 1;
 sd::LongType  len = indices->lengthOf();
 classesRangesBegs.assign(len);
 classesRangesLens.assign(zero);
 dim3 dims= getLaunchDims("segmentSqrtN");
 fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
 output->nullify();
 if (input->isVector()  || input->isScalar()) {
   unsortedSegmentSqrtNLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(), indices->dataBuffer()->specialAsT<I>(),
       indices->specialShapeInfo(), begins, lengths, numOfClasses, output->dataBuffer()->specialAsT<T>(),
       output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentSqrtNLinearKernel failed");

 } else {
   output->nullify();
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   dims.x = input->sizeAt(0);
   segmentSqrtNTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(), inputTads, inputTadOffsets,
       indices->dataBuffer()->specialAsT<I>(), begins, lengths, numOfClasses, output->specialBuffer(),
       output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentSqrtNTadKernel failed");

   delete dimensions;
 }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentSqrtNFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices});
 BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentSqrtNFunctor_,
                       (context, input, indices, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices});
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentSqrtNBPLinearKernel(void* inputBuf, LongType const* inputShape, void* eps,
                                                 LongType const* epsShape, void* indicesBuf,
                                                 LongType const* indicesShape, LongType* lengths, void* outputBuf,
                                                 LongType const* outputShape) {
 __shared__ T* x;
 __shared__ T* gradIn;
 __shared__ T* gradOut;
 __shared__ I* y;
 __shared__ T* z;
 __shared__ LongType xLen, gradLen;

 // Cache shape information
 __shared__ sd::LongType inputRank, outputRank, indicesRank, epsRank;
 __shared__ const sd::LongType* inputShapePtr;
 __shared__ const sd::LongType* outputShapePtr;
 __shared__ const sd::LongType* indicesShapePtr;
 __shared__ const sd::LongType* epsShapePtr;
 __shared__ const sd::LongType* inputStridePtr;
 __shared__ const sd::LongType* outputStridePtr;
 __shared__ const sd::LongType* indicesStridePtr;
 __shared__ const sd::LongType* epsStridePtr;

 if (threadIdx.x == 0) {
   xLen = shape::length(inputShape);
   x = reinterpret_cast<T*>(inputBuf);
   y = reinterpret_cast<I*>(indicesBuf);
   z = reinterpret_cast<T*>(outputBuf);
   gradOut = reinterpret_cast<T*>(eps);
   gradLen = shape::length(epsShape);

   // Cache all shape information
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

 LongType zCoords[SD_MAX_RANK];
 LongType xCoords[SD_MAX_RANK];
 LongType yCoords[SD_MAX_RANK];
 LongType zOffset;
 LongType xOffset;
 LongType yOffset;
 LongType gradOffsetO;

 for (auto e = start; e < xLen; e += step) {
   INDEX2COORDS(e, outputRank, outputShapePtr, zCoords);
   COORDS2INDEX(outputRank, outputStridePtr, zCoords, zOffset);
   INDEX2COORDS(e, inputRank, inputShapePtr, xCoords);
   COORDS2INDEX(inputRank, inputStridePtr, xCoords, xOffset);
   INDEX2COORDS(e, indicesRank, indicesShapePtr, yCoords);
   COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yOffset);
   auto classIndex = y[yOffset];
   INDEX2COORDS(classIndex, epsRank, epsShapePtr, zCoords);
   COORDS2INDEX(epsRank, epsStridePtr, zCoords, gradOffsetO);

   z[zOffset] = T(gradOut[gradOffsetO] / math::sd_sqrt<LongType, float>(lengths[classIndex]));
 }
}

template <typename T, typename I>
static SD_KERNEL void segmentSqrtNBPTadKernel(void* inputBuf, LongType const* inputShape, void* eps,
                                              LongType const* epsShape, void* indicesBuf, LongType const* indicesShape,
                                              LongType* lengths, void* outputBuf, LongType const* outputShape,
                                              LongType const* inputTad, LongType const* inputOffsets,
                                              LongType const* gradOutTad, LongType const* gradOutOffsets,
                                              LongType const* outTad, LongType const* outOffsets) {
 __shared__ T* x;
 __shared__ T* gradOut;
 __shared__ I* y;
 __shared__ T* z;
 __shared__ LongType xLen, yLen, gradLen, currentLen;

 // Cache shape information
 __shared__ sd::LongType outTadRank, gradOutTadRank;
 __shared__ const sd::LongType* outTadShapePtr;
 __shared__ const sd::LongType* gradOutTadShapePtr;
 __shared__ const sd::LongType* outTadStridePtr;
 __shared__ const sd::LongType* gradOutTadStridePtr;

 if (threadIdx.x == 0) {
   xLen = shape::length(inputShape);
   x = reinterpret_cast<T*>(inputBuf);
   y = reinterpret_cast<I*>(indicesBuf);
   z = reinterpret_cast<T*>(outputBuf);
   yLen = shape::length(indicesShape);
   gradOut = reinterpret_cast<T*>(eps);
   gradLen = shape::length(epsShape);
   currentLen = shape::length(outTad);

   // Cache TAD shape information
   outTadRank = shape::rank(outTad);
   gradOutTadRank = shape::rank(gradOutTad);
   outTadShapePtr = shape::shapeOf(outTad);
   gradOutTadShapePtr = shape::shapeOf(gradOutTad);
   outTadStridePtr = shape::stride(outTad);
   gradOutTadStridePtr = shape::stride(gradOutTad);
 }
 __syncthreads();

 for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
   auto segment = y[i];
   T* currentOut = z + outOffsets[i];
   T* outGrad = gradOut + gradOutOffsets[segment];

   for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
     LongType zCoords[SD_MAX_RANK];
     LongType gradCoords[SD_MAX_RANK];
     LongType zIndex;
     LongType gradIndex;

     INDEX2COORDS(e, outTadRank, outTadShapePtr, zCoords);
     COORDS2INDEX(outTadRank, outTadStridePtr, zCoords, zIndex);
     INDEX2COORDS(e, gradOutTadRank, gradOutTadShapePtr, gradCoords);
     COORDS2INDEX(gradOutTadRank, gradOutTadStridePtr, gradCoords, gradIndex);

     if (lengths[segment] > 0)
       currentOut[zIndex] = T(outGrad[gradIndex] / math::sd_sqrt<LongType, float>(lengths[segment]));
   }
 }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static Status unsortedSegmentSqrtNFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                            NDArray* gradOut,
                                            LongType numOfClasses, NDArray* output) {
 auto stream = context->getCudaStream();
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 auto numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
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

 if (input->isVector()  || input->isScalar()) {
   LongType loop_size = input->lengthOf();
   auto numOfClasses = gradOut->lengthOf();
   segmentSqrtNBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
       output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentSqrtNBPLinearKernel failed");

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
   dim3 segmentBpTad2 = segmentBpTad(indices->lengthOf(),input->lengthOf());

   segmentSqrtNBPTadKernel<T, I><<<segmentBpTad2.y, segmentBpTad2.x, segmentBpTad2.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
       output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads,
       outputTadOffsets);
   sd::DebugHelper::checkErrorCode(stream, "segmentSqrtNBPTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});

 return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentSqrtNFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                    LongType numOfClasses, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentSqrtNFunctorBP_,
                       (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
