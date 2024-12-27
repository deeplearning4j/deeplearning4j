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
// Segment ops linear kernels
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentMeanLinearKernel(void* input, LongType const* inputShape, LongType* indices,
                                              LongType* lengths, LongType numOfClasses, void* output,
                                              LongType const* outputShape) {
  __shared__ T* val;
  __shared__ LongType xLen, zLen, zIndex;
  __shared__ T* x;
  __shared__ T* z;
  __shared__ LongType threadsPerSegment, start, finish;

  // Cache shape information
  __shared__ sd::LongType inputRank, outputRank;
  __shared__ const sd::LongType* inputShapePtr;
  __shared__ const sd::LongType* outputShapePtr;
  __shared__ const sd::LongType* inputStridePtr;
  __shared__ const sd::LongType* outputStridePtr;

  auto segment = blockIdx.x;
  if (threadIdx.x == 0) {
    x = reinterpret_cast<T*>(input);
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
      LongType outputCoords[SD_MAX_RANK];
      LongType inputCoords[SD_MAX_RANK];
      LongType xOffset;
      LongType zOffset;

      INDEX2COORDS(segment, outputRank, outputShapePtr, outputCoords);
      COORDS2INDEX(outputRank, outputStridePtr, outputCoords, zIndex);
      start = indices[segment];
      finish = start + lengths[segment];
      INDEX2COORDS(start, inputRank, inputShapePtr, inputCoords);
      COORDS2INDEX(inputRank, inputStridePtr, inputCoords, xOffset);
      if (lengths[segment] > 0)
        z[zIndex] = T(x[xOffset] / T(lengths[segment]));
      else
        z[zIndex] = 0;
      val[segment] = z[zIndex];
    }
  }
  __syncthreads();

  for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
    LongType inputCoords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(e, inputRank, inputShapePtr, inputCoords);
    COORDS2INDEX(inputRank, inputStridePtr, inputCoords, xOffset);
    math::atomics::sd_atomicAdd(&z[zIndex], T(x[xOffset] / static_cast<T>(lengths[segment])));
  }
}

template <typename T, typename I>
static SD_KERNEL void unsortedSegmentMeanLinearKernel(void* input, LongType const* inputShape, void* indices,
                                                      LongType const* indicesShape, LongType* starts, LongType* lengths,
                                                      LongType numOfClasses, void* output,
                                                      LongType const* outputShape) {
  __shared__ LongType xLen, zLen, zIndex;
  __shared__ T* x;
  __shared__ T* z;
  __shared__ I* y;

  // Cache shape information
  __shared__ sd::LongType inputRank, outputRank, indicesRank;
  __shared__ const sd::LongType* inputShapePtr;
  __shared__ const sd::LongType* outputShapePtr;
  __shared__ const sd::LongType* indicesShapePtr;
  __shared__ const sd::LongType* inputStridePtr;
  __shared__ const sd::LongType* outputStridePtr;
  __shared__ const sd::LongType* indicesStridePtr;

  auto segment = blockIdx.x;
  if (threadIdx.x == 0) {
    x = reinterpret_cast<T*>(input);
    z = reinterpret_cast<T*>(output);
    y = reinterpret_cast<I*>(indices);
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

    LongType outputCoords[SD_MAX_RANK];
    LongType inputCoords[SD_MAX_RANK];
    LongType xOffset;
    LongType zOffset;

    INDEX2COORDS(segment, outputRank, outputShapePtr, outputCoords);
    COORDS2INDEX(outputRank, outputStridePtr, outputCoords, zIndex);
    INDEX2COORDS(starts[segment], inputRank, inputShapePtr, inputCoords);
    COORDS2INDEX(inputRank, inputStridePtr, inputCoords, xOffset);

    if (lengths[segment] > 0)
      z[zIndex] = T(x[xOffset] / T(lengths[segment]));
    else
      z[zIndex] = 0;
  }
  __syncthreads();

  if (lengths[segment] > 0) {
    for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
      LongType inputCoords[SD_MAX_RANK];
      LongType xOffset;
      LongType yIndex;

      INDEX2COORDS(e, inputRank, inputShapePtr, inputCoords);
      COORDS2INDEX(inputRank, inputStridePtr, inputCoords, xOffset);
      INDEX2COORDS(e, indicesRank, indicesShapePtr, inputCoords);
      COORDS2INDEX(indicesRank, indicesStridePtr, inputCoords, yIndex);

      if (y[yIndex] == segment && e != starts[segment]) {
        math::atomics::sd_atomicAdd(&z[zIndex], T(x[xOffset] / T(lengths[segment])));
      }
    }
  }
}

template <typename T, typename I>
static SD_KERNEL void segmentMeanTadKernel(void* inputBuf, LongType const* inputShape, LongType const* inputTads,
                                           LongType const* inputTadOffsets,
                                           I* indices, LongType* starts,
                                           LongType* lengths, LongType numOfClasses, void* outputBuf,
                                           LongType const* outputShape, LongType const* outputTads,
                                           LongType const* outputTadOffsets, LongType indicesLen) {
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
    auto x = reinterpret_cast<T*>(inputBuf) + inputTadOffsets[idx];
    if (blockIdx.x == start) {
      for (auto e = threadIdx.x; e < len; e += blockDim.x) {
        LongType xCoords[SD_MAX_RANK];
        LongType zCoords[SD_MAX_RANK];
        LongType xIndex;
        LongType zIndex;

        INDEX2COORDS(e, inputTadRank, inputTadShapePtr, xCoords);
        COORDS2INDEX(inputTadRank, inputTadStridePtr, xCoords, xIndex);
        INDEX2COORDS(e, outputTadRank, outputTadShapePtr, zCoords);
        COORDS2INDEX(outputTadRank, outputTadStridePtr, zCoords, zIndex);

        math::atomics::sd_atomicAdd(&z[zIndex], T(x[xIndex] / lengths[segment]));
      }
    } else {
      for (auto e = threadIdx.x; e < len; e += blockDim.x) {
        LongType xCoords[SD_MAX_RANK];
        LongType zCoords[SD_MAX_RANK];
        LongType xIndex;
        LongType zIndex;

        INDEX2COORDS(e, inputTadRank, inputTadShapePtr, xCoords);
        COORDS2INDEX(inputTadRank, inputTadStridePtr, xCoords, xIndex);
        INDEX2COORDS(e, outputTadRank, outputTadShapePtr, zCoords);
        COORDS2INDEX(outputTadRank, outputTadStridePtr, zCoords, zIndex);

        if (lengths[segment]) math::atomics::sd_atomicAdd(&z[zIndex], T(x[xIndex] / lengths[segment]));
      }
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
// segment mean
template <typename T, typename I>
static void segmentMeanFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
 auto stream = context->getCudaStream();
 LongType numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
 NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
 NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);

 int zero2 = 0;
 sd::LongType len = indices->lengthOf();
 classesRangesBegs.assign(len);
 classesRangesLens.assign(zero2);
 NDArray::prepareSpecialUse({output}, {input, indices});
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
 fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);

 if (input->isVector()  || input->isScalar()) {
   dim3 launchDims = segmentDims(numClasses,input->lengthOf());
   segmentMeanLinearKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(),
       output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentMeanLinearKernel failed");

 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   dim3 launchDims = segmentTad(input->sizeAt(0));
   segmentMeanTadKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
       reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(),
       output->specialShapeInfo(), outputTads, outputTadOffsets,indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentMeanTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices});
}
// -------------------------------------------------------------------------------------------------------------- //
void segmentMeanFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices});
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segmentMeanFunctor_, (context, input, indices, output),
                       SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static void unsortedSegmentMeanFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
 auto stream = context->getCudaStream();

 NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
 NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);

 int zero2 = 0;
 sd::LongType len = indices->lengthOf();
 classesRangesBegs.assign(len);
 classesRangesLens.assign(zero2);
 dim3 dims = getFillUpSegmentsDims(numOfClasses, indices->lengthOf());
 fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

 if (input->isVector()  || input->isScalar()) {
   unsortedSegmentMeanLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentMeanLinearKernel failed");

 } else {
   LongType zero = 0;
   output->assign(zero);
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   LongType const* inputTads = packX->specialShapeInfo();
   LongType const* inputTadOffsets = packX->specialOffsets();
   LongType const* outputTads = packZ->specialShapeInfo();
   LongType const* outputTadOffsets = packZ->specialOffsets();
   dims.x = input->sizeAt(0);
   segmentMeanTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
       reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
       output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentMeanTadKernel failed");

   delete dimensions;
 }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentMeanFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                               NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices});
 BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMeanFunctor_,
                       (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices});
}

template <typename T, typename I>
static SD_KERNEL void segmentMeanBPLinearKernel(void* inputBuf, LongType const* inputShape, void* eps,
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

 for (auto e = start; e < xLen; e += step) {
   LongType zOffset, xOffset, yOffset, gradOffsetO;
   sd::LongType zCoords[SD_MAX_RANK], xCoords[SD_MAX_RANK], yCoords[SD_MAX_RANK], gradCoords[SD_MAX_RANK];

   INDEX2COORDS(e, outputRank, outputShapePtr, zCoords);
   COORDS2INDEX(outputRank, outputStridePtr, zCoords, zOffset);

   INDEX2COORDS(e, inputRank, inputShapePtr, xCoords);
   COORDS2INDEX(inputRank, inputStridePtr, xCoords, xOffset);

   INDEX2COORDS(e, indicesRank, indicesShapePtr, yCoords);
   COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yOffset);

   auto classIndex = y[yOffset];

   INDEX2COORDS(classIndex, epsRank, epsShapePtr, gradCoords);
   COORDS2INDEX(epsRank, epsStridePtr, gradCoords, gradOffsetO);

   z[zOffset] = T(gradOut[gradOffsetO] / float(lengths[classIndex]));
 }
}

template <typename T, typename I>
static SD_KERNEL void segmentMeanBPTadKernel(void* inputBuf, LongType const* inputShape, void* eps,
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
     sd::LongType zCoords[SD_MAX_RANK];
     sd::LongType gradCoords[SD_MAX_RANK];
     sd::LongType zIndex;
     sd::LongType gradIndex;

     INDEX2COORDS(e, outTadRank, outTadShapePtr, zCoords);
     COORDS2INDEX(outTadRank, outTadStridePtr, zCoords, zIndex);
     INDEX2COORDS(e, gradOutTadRank, gradOutTadShapePtr, gradCoords);
     COORDS2INDEX(gradOutTadRank, gradOutTadStridePtr, gradCoords, gradIndex);

     if (lengths[segment] > 0) currentOut[zIndex] = T(outGrad[gradIndex] / float(lengths[segment]));
   }
 }
}
// -------------------------------------------------------------------------------------------------------------- //
// backrop for mean
template <typename T, typename I>
Status segmentMeanFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                            NDArray* output) {
 auto stream = context->getCudaStream();
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 auto numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
 NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
 NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);
 sd::LongType zero2 = 0;
 sd::LongType len = indices->lengthOf();
 classesRangesBegs.assign(zero2);
 classesRangesLens.assign(len);
 fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

 if (input->isVector()  || input->isScalar()) {
   LongType loop_size = input->lengthOf();
   auto numOfClasses = gradOut->lengthOf();  // indices->e<sd::LongType>(loop_size - 1);
   dim3 segmentBpDims2 = segmentBpDims(gradOut->lengthOf(),input->lengthOf());
   segmentMeanBPLinearKernel<T, I><<<segmentBpDims2.y, segmentBpDims2.x, segmentBpDims2.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
       output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentMeanBPLinearKernel failed");

 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
   LongType const* inputTads = packX->specialShapeInfo();
   LongType const* inputTadOffsets = packX->specialOffsets();
   LongType const* outputTads = packZ->specialShapeInfo();
   LongType const* outputTadOffsets = packZ->specialOffsets();
   LongType const* gradOutTads = packGradOut->specialShapeInfo();
   LongType const* gradOutTadOffsets = packGradOut->specialOffsets();
   dim3 segmentBpTad2 = segmentBpTad(indices->lengthOf(),input->lengthOf());

   segmentMeanBPTadKernel<T, I><<<segmentBpTad2.y, segmentBpTad2.x, segmentBpTad2.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
       output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads,
       outputTadOffsets);
   sd::DebugHelper::checkErrorCode(stream, "segmentMeanBPTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
 return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
// segmen mean bp main
Status segmentMeanFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                           NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentMeanFunctorBP_,
                       (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static Status unsortedSegmentMeanFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                           NDArray* gradOut,
                                           LongType numOfClasses, NDArray* output) {
 auto stream = context->getCudaStream();
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 auto numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
 NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
 NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);

 sd::LongType zero2 = 0;
 sd::LongType len = indices->lengthOf();
 classesRangesBegs.assign(zero2);
 classesRangesLens.assign(len);
 fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

 if (input->isVector()  || input->isScalar()) {
   LongType loop_size = input->lengthOf();
   auto numOfClasses = gradOut->lengthOf();
   dim3 segmentBpDims2 = segmentBpDims(gradOut->lengthOf(),input->lengthOf());
   segmentMeanBPLinearKernel<T, I><<<segmentBpDims2.y,segmentBpDims2.x,segmentBpDims2.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
       output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentMeanBPLinearKernel failed");


 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1, &zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);

   auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
   LongType const* inputTads = packX->specialShapeInfo();
   LongType const* inputTadOffsets = packX->specialOffsets();
   LongType const* outputTads = packZ->specialShapeInfo();
   LongType const* outputTadOffsets = packZ->specialOffsets();
   LongType const* gradOutTads = packGradOut->specialShapeInfo();
   LongType const* gradOutTadOffsets = packGradOut->specialOffsets();
   dim3 segmentBpTad2 = segmentBpTad(indices->lengthOf(),input->lengthOf());

   segmentMeanBPTadKernel<T, I><<<segmentBpTad2.y,segmentBpTad2.x, segmentBpTad2.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
       output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads,
       outputTadOffsets);
   sd::DebugHelper::checkErrorCode(stream, "segmentMeanBPTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
 return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentMeanFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                   LongType numOfClasses, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentMeanFunctorBP_,
                       (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
