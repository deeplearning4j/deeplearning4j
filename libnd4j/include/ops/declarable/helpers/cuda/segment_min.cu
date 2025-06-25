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
static SD_KERNEL void segmentMinLinearKernel(const void* input, const LongType* inputShape, LongType* starts,
                                             LongType* lengths, LongType numOfClasses, void* output,
                                             const LongType* outputShape) {
  __shared__ T* val;
  __shared__ LongType xLen, zLen, zIndex;
  __shared__ const T* x;
  __shared__ T* z;
  __shared__ LongType threadsPerSegment, start, finish;

  // Cache shape information
  __shared__ sd::LongType inputRank, outputRank;
  __shared__ const sd::LongType* inputShapePtr;
  __shared__ const sd::LongType* outputShapePtr;
  __shared__ const sd::LongType* inputStridePtr;
  __shared__ const sd::LongType* outputStridePtr;

  auto segment = blockIdx.x;
  if(blockIdx.x >= numOfClasses)
    return;

  if (threadIdx.x == 0) {
    x = reinterpret_cast<const T*>(input);
    z = reinterpret_cast<T*>(output);
    extern __shared__ unsigned char shmem[];
    val = reinterpret_cast<T*>(shmem);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    // Cache shape information
    inputRank = shape::rank(inputShape);
    outputRank = shape::rank(outputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    inputStridePtr = shape::stride(inputShape);
    outputStridePtr = shape::stride(outputShape);

    if (segment < numOfClasses) {
      LongType zCoords[SD_MAX_RANK];
      INDEX2COORDS(segment, outputRank, outputShapePtr, zCoords);
      COORDS2INDEX(outputRank, outputStridePtr, zCoords, zIndex);
      if(zIndex >= zLen)
        return;
      start = starts[segment];
      finish = start + lengths[segment];
      LongType startCoords[SD_MAX_RANK];
      LongType startIndex;
      INDEX2COORDS(start, inputRank, inputShapePtr, startCoords);
      COORDS2INDEX(inputRank, inputStridePtr, startCoords, startIndex);
      z[zIndex] = x[startIndex];
      val[segment] = z[zIndex];
    }
  }
  __syncthreads();

  for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
    LongType eCoords[SD_MAX_RANK];
    LongType eIndex;
    INDEX2COORDS(e, inputRank, inputShapePtr, eCoords);
    COORDS2INDEX(inputRank, inputStridePtr, eCoords, eIndex);
    if (eIndex >= xLen) return;
    math::atomics::sd_atomicMin(&z[zIndex], x[eIndex]);
  }
}

template <typename T, typename I>
static SD_KERNEL void unsortedSegmentMinLinearKernel(const void* input, const LongType* inputShape,
                                                     const void* indices, const LongType* indicesShape, LongType* starts, LongType* lengths,
                                                     LongType numOfClasses, void* output,
                                                     const LongType* outputShape) {
  __shared__ T* val;
  __shared__ LongType xLen, zLen, segment, zIndex;
  __shared__ const T* x;
  __shared__ T* z;
  __shared__ const I* y;

  // Cache shape information
  __shared__ sd::LongType inputRank, outputRank, indicesRank;
  __shared__ const sd::LongType* inputShapePtr;
  __shared__ const sd::LongType* outputShapePtr;
  __shared__ const sd::LongType* indicesShapePtr;
  __shared__ const sd::LongType* inputStridePtr;
  __shared__ const sd::LongType* outputStridePtr;
  __shared__ const sd::LongType* indicesStridePtr;

  if (threadIdx.x == 0) {
    segment = blockIdx.x;
    x = reinterpret_cast<const T*>(input);
    z = reinterpret_cast<T*>(output);
    y = reinterpret_cast<const I*>(indices);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    // Cache shape information
    inputRank = shape::rank(inputShape);
    outputRank = shape::rank(outputShape);
    indicesRank = shape::rank(indicesShape);
    inputShapePtr = shape::shapeOf(inputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    indicesShapePtr = shape::shapeOf(indicesShape);
    inputStridePtr = shape::stride(inputShape);
    outputStridePtr = shape::stride(outputShape);
    indicesStridePtr = shape::stride(indicesShape);

    LongType zCoords[SD_MAX_RANK];
    INDEX2COORDS(segment, outputRank, outputShapePtr, zCoords);
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zIndex);
    if (lengths[segment] > 0) {
      LongType startCoords[SD_MAX_RANK];
      LongType startIndex;
      INDEX2COORDS(starts[segment], inputRank, inputShapePtr, startCoords);
      COORDS2INDEX(inputRank, inputStridePtr, startCoords, startIndex);
      z[zIndex] = x[startIndex];
    } else {
      z[zIndex] = DataTypeUtils::max<T>();
    }
  }
  __syncthreads();

  if (lengths[segment] > 0) {
    for (auto e = threadIdx.x + 1; e < xLen; e += blockDim.x) {
      LongType eCoords[SD_MAX_RANK];
      LongType eIndex;
      INDEX2COORDS(e, inputRank, inputShapePtr, eCoords);
      COORDS2INDEX(inputRank, inputStridePtr, eCoords, eIndex);

      LongType yCoords[SD_MAX_RANK];
      LongType yIndex;
      INDEX2COORDS(e, indicesRank, indicesShapePtr, yCoords);
      COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yIndex);

      if (y[yIndex] == segment) {
        math::atomics::sd_atomicMin(&z[zIndex], x[eIndex]);
      }
    }
  }
}

template <typename T, typename I>
static SD_KERNEL void segmentMinTadKernel(const void* inputBuf, const LongType* inputShape,
                                          const LongType* inputTads, const LongType* inputTadOffsets,
                                          I* indices, LongType* starts,
                                          LongType* lengths, LongType numOfClasses, void* outputBuf, const LongType* outputShape,
                                          const LongType* outputTads, const LongType* outputTadOffsets, LongType indicesLen) {
  __shared__ T* val;
  __shared__ LongType len, zIndex, total;
  __shared__ T* z;
  __shared__ int threadsPerSegment, start, finish;

  // Cache shape information
  __shared__ sd::LongType inputTadRank, outputTadRank;
  __shared__ const sd::LongType* inputTadShapePtr;
  __shared__ const sd::LongType* outputTadShapePtr;
  __shared__ const sd::LongType* inputTadStridePtr;
  __shared__ const sd::LongType* outputTadStridePtr;

  if(blockIdx.x >= indicesLen)
    return;

  auto segment = indices[blockIdx.x];

  if (threadIdx.x == 0) {
    z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
    len = shape::length(inputTads);
    start = starts[segment];
    finish = start + lengths[segment];
    total = shape::sizeAt(inputShape, 0);

    // Cache TAD shape information
    inputTadRank = shape::rank(inputTads);
    outputTadRank = shape::rank(outputTads);
    inputTadShapePtr = shape::shapeOf(inputTads);
    outputTadShapePtr = shape::shapeOf(outputTads);
    inputTadStridePtr = shape::stride(inputTads);
    outputTadStridePtr = shape::stride(outputTads);
  }
  __syncthreads();

  auto idx = blockIdx.x;
  if (blockIdx.x <= total) {
    auto x = reinterpret_cast<const T*>(inputBuf) + inputTadOffsets[idx];
    LongType xCoords[SD_MAX_RANK];
    LongType zCoords[SD_MAX_RANK];
    LongType xOffset;
    LongType zOffset;

    for (auto e = threadIdx.x; e < len; e += blockDim.x) {
      INDEX2COORDS(e, inputTadRank, inputTadShapePtr, xCoords);
      COORDS2INDEX(inputTadRank, inputTadStridePtr, xCoords, xOffset);
      INDEX2COORDS(e, outputTadRank, outputTadShapePtr, zCoords);
      COORDS2INDEX(outputTadRank, outputTadStridePtr, zCoords, zOffset);
      math::atomics::sd_atomicMin(&z[zOffset], x[xOffset]);
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
// segmen min
template <typename T, typename I>
static void segmentMinFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
 auto stream = context->getCudaStream();
 LongType numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
 auto classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
 auto classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);
 T val = DataTypeUtils::infOrMax<T>();
 output->assign(val);
 sd::LongType zero2 = 0;
 sd::LongType len = indices->lengthOf();
 classesRangesBegs.assign(zero2);
 classesRangesLens.assign(len);
 fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
 NDArray::prepareSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
 if (input->isVector()  || input->isScalar()) {
   dim3 launchDims = segmentDims(numClasses,input->lengthOf());
   segmentMinLinearKernel<T, I><<<launchDims.y,launchDims.x, launchDims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(),
       output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentMinLinearKernel failed");

 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   dim3 launchDims = segmentTad(input->sizeAt(0));
   segmentMinTadKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
       reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(),
       output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentMinTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});
}
// -------------------------------------------------------------------------------------------------------------- //
void segmentMinFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices});
 output->nullify();
#if SD_IS_PAIR_TYPE_COMPILED(input->dataType(),indices->dataType())
 BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentMinFunctor_, (context, input, indices, output),
                       SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
#endif
 NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void unsortedSegmentMinFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
 auto stream = context->getCudaStream();
 NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
 NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
 T val = DataTypeUtils::infOrMax<T>();
 sd::LongType  len = indices->lengthOf();
 output->assign(val);
 sd::LongType  zero = 0;
 classesRangesBegs.assign(len);
 classesRangesLens.assign(zero);
 dim3 dims = getFillUpSegmentsDims(numOfClasses, indices->lengthOf());
 fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
 NDArray::prepareSpecialUse({output}, {input, indices});
 if (input->isVector()  || input->isScalar()) {
   unsortedSegmentMinLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentMinLinearKernel failed");

 } else {
   T val = DataTypeUtils::max<T>();
   output->assign(val);
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   dims.x = input->sizeAt(0);
   segmentMinTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
       reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
       output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentMinTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices});
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentMinFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                              NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices});
 output->nullify();
#if SD_IS_PAIR_TYPE_COMPILED(input->dataType(),indices->dataType())
 BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMinFunctor_,
                       (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
#endif
 NDArray::registerSpecialUse({output}, {input, indices});
}

template <typename T, typename I>
static SD_KERNEL void segmentMinBPLinearKernel(const void* inputBuf, const LongType* inputShape,
                                               void* forwardOutput, const LongType* forwardShape, void* eps,
                                               const LongType* epsShape, const void* indicesBuf,
                                               const LongType* indicesShape, void* outputBuf,
                                               const LongType* outputShape) {
 __shared__ const T* x;
 __shared__ T* gradIn;
 __shared__ T* gradOut;
 __shared__ const I* y;
 __shared__ T* z;
 __shared__ LongType xLen, gradLen;

 // Cache shape information
 __shared__ sd::LongType inputRank, outputRank, indicesRank, forwardRank, epsRank;
 __shared__ const sd::LongType* inputShapePtr;
 __shared__ const sd::LongType* outputShapePtr;
 __shared__ const sd::LongType* indicesShapePtr;
 __shared__ const sd::LongType* forwardShapePtr;
 __shared__ const sd::LongType* epsShapePtr;
 __shared__ const sd::LongType* inputStridePtr;
 __shared__ const sd::LongType* outputStridePtr;
 __shared__ const sd::LongType* indicesStridePtr;
 __shared__ const sd::LongType* forwardStridePtr;
 __shared__ const sd::LongType* epsStridePtr;

 if (threadIdx.x == 0) {
   xLen = shape::length(inputShape);
   x = reinterpret_cast<const T*>(inputBuf);
   y = reinterpret_cast<const I*>(indicesBuf);
   z = reinterpret_cast<T*>(outputBuf);
   gradIn = reinterpret_cast<T*>(forwardOutput);
   gradOut = reinterpret_cast<T*>(eps);
   gradLen = shape::length(epsShape);

   // Cache all shape information
   inputRank = shape::rank(inputShape);
   outputRank = shape::rank(outputShape);
   indicesRank = shape::rank(indicesShape);
   forwardRank = shape::rank(forwardShape);
   epsRank = shape::rank(epsShape);

   inputShapePtr = shape::shapeOf(inputShape);
   outputShapePtr = shape::shapeOf(outputShape);
   indicesShapePtr = shape::shapeOf(indicesShape);
   forwardShapePtr = shape::shapeOf(forwardShape);
   epsShapePtr = shape::shapeOf(epsShape);

   inputStridePtr = shape::stride(inputShape);
   outputStridePtr = shape::stride(outputShape);
   indicesStridePtr = shape::stride(indicesShape);
   forwardStridePtr = shape::stride(forwardShape);
   epsStridePtr = shape::stride(epsShape);
 }
 __syncthreads();

 auto start = blockIdx.x * blockDim.x + threadIdx.x;
 auto step = gridDim.x * blockDim.x;

 for (auto e = start; e < xLen; e += step) {
   LongType zCoords[SD_MAX_RANK];
   LongType xCoords[SD_MAX_RANK];
   LongType yCoords[SD_MAX_RANK];
   LongType gradICoords[SD_MAX_RANK];
   LongType gradOCoords[SD_MAX_RANK];
   LongType zOffset;
   LongType xOffset;
   LongType yOffset;
   LongType gradOffsetI;
   LongType gradOffsetO;

   INDEX2COORDS(e, outputRank, outputShapePtr, zCoords);
   COORDS2INDEX(outputRank, outputStridePtr, zCoords, zOffset);
   INDEX2COORDS(e, inputRank, inputShapePtr, xCoords);
   COORDS2INDEX(inputRank, inputStridePtr, xCoords, xOffset);
   INDEX2COORDS(e, indicesRank, indicesShapePtr, yCoords);
   COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yOffset);
   auto classIndex = y[yOffset];
   INDEX2COORDS(classIndex, forwardRank, forwardShapePtr, gradICoords);
   COORDS2INDEX(forwardRank, forwardStridePtr, gradICoords, gradOffsetI);
   INDEX2COORDS(classIndex, epsRank, epsShapePtr, gradOCoords);
   COORDS2INDEX(epsRank, epsStridePtr, gradOCoords, gradOffsetO);

   if (math::sd_abs<T, T>(gradIn[gradOffsetI] - x[xOffset]) <= T(1.e-6)) {
     z[zOffset] = gradOut[gradOffsetO];
   }
 }
}

template <typename T, typename I>
static SD_KERNEL void segmentMinBPTadKernel(const void* inputBuf, const LongType* inputShape, void* forwardOutput,
                                            const LongType* forwardShape, void* eps, const LongType* epsShape,
                                            const void* indicesBuf, const LongType* indicesShape, void* outputBuf,
                                            const LongType* outputShape, const LongType* inputTad,
                                            const LongType* inputOffsets, const LongType* gradInTad,
                                            const LongType* gradInOffsets, const LongType* gradOutTad,
                                            const LongType* gradOutOffsets, const LongType* outTad,
                                            const LongType* outOffsets) {
 __shared__ const T* x;
 __shared__ T* gradIn;
 __shared__ T* gradOut;
 __shared__ const I* y;
 __shared__ T* z;
 __shared__ LongType xLen, yLen, gradLen, currentLen;

 // Cache shape information
 __shared__ sd::LongType indicesRank;
 __shared__ const sd::LongType* indicesShapePtr;
 __shared__ const sd::LongType* indicesStridePtr;

 if (threadIdx.x == 0) {
   xLen = shape::length(inputShape);
   x = reinterpret_cast<const T*>(inputBuf);
   y = reinterpret_cast<const I*>(indicesBuf);
   z = reinterpret_cast<T*>(outputBuf);
   yLen = shape::length(indicesShape);
   gradOut = reinterpret_cast<T*>(eps);
   gradIn = reinterpret_cast<T*>(forwardOutput);
   gradLen = shape::length(epsShape);
   currentLen = shape::length(outTad);

   // Cache indices shape information (only needed for segment calculation)
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
   auto current = x + inputOffsets[i];
   auto currentOut = z + outOffsets[i];
   auto in = gradIn + gradInOffsets[segment];
   auto outGrad = gradOut + gradOutOffsets[segment];

   for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
     if (math::sd_abs<T,T>(in[e] - current[e]) <= T(1.e-6)) currentOut[e] = outGrad[e];
   }
 }
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
Status segmentMinFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                           NDArray* output) {

 // if input is a vector: (as if in doc sample)
 auto stream = context->getCudaStream();
 auto outShape = gradOut->getShapeAsVector();
 NDArray tempRes(gradOut->ordering(), outShape, DataTypeUtils::fromT<T>(),
                 context);
 segmentMinFunctor_<T, I>(context, input, indices, &tempRes);
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
 if (input->isVector()  || input->isScalar()) {
   LongType loop_size = input->lengthOf();
   auto numOfClasses = gradOut->lengthOf();

   segmentMinBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
       gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       output->specialBuffer(), output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentMinBPLinearKernel failed");


 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto packGradIn = ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
   auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   auto gradInTads = packGradIn->specialShapeInfo();
   auto gradInTadOffsets = packGradIn->specialOffsets();
   auto gradOutTads = packGradOut->specialShapeInfo();
   auto gradOutTadOffsets = packGradOut->specialOffsets();

   segmentMinBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
       gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
       gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
   sd::DebugHelper::checkErrorCode(stream, "segmentMinBPTadKernel failed");

 }
 NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
 return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
// segmen min
Status segmentMinFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                          NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
#if SD_IS_PAIR_TYPE_COMPILED(output->dataType(),indices->dataType())
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentMinFunctorBP_,
                       (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
#endif
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

template <typename T, typename I>
static Status unsortedSegmentMinFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                          NDArray* gradOut,
                                          LongType numOfClasses, NDArray* output) {
 // if input is a vector: (as if in doc sample)
 auto stream = context->getCudaStream();
 auto outShape = gradOut->getShapeAsVector();

 NDArray tempRes(gradOut->ordering(), outShape, DataTypeUtils::fromT<T>(),
                 context);
 unsortedSegmentMinFunctor_<T, I>(context, input, indices, numOfClasses, &tempRes);
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
 if (input->isVector()  || input->isScalar()) {
   LongType loop_size = input->lengthOf();
   auto numOfClasses = gradOut->lengthOf();
   segmentMinBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
       gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       output->specialBuffer(), output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentMinBPLinearKernel failed");

 } else {
   LongType zero = 0;

   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto packGradIn = ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
   auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   auto gradInTads = packGradIn->specialShapeInfo();
   auto gradInTadOffsets = packGradIn->specialOffsets();
   auto gradOutTads = packGradOut->specialShapeInfo();
   auto gradOutTadOffsets = packGradOut->specialOffsets();

   segmentMinBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
       gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
       gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
   sd::DebugHelper::checkErrorCode(stream, "segmentMinBPTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
 return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentMinFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                  LongType numOfClasses, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
#if SD_IS_PAIR_TYPE_COMPILED(output->dataType(),indices->dataType())
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentMinFunctorBP_,
                       (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
#endif
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
