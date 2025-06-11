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
static SD_KERNEL void segmentMaxLinearKernel(void* input, LongType const* inputShape, LongType* starts,
                                             LongType* lengths, LongType numOfClasses, void* output,
                                             LongType const* outputShape) {
  __shared__ T* val;
  __shared__ LongType xLen, zLen, zIndex;
  __shared__ T* x;
  __shared__ T* z;
  __shared__ LongType threadsPerSegment, start, finish;

  // Cache shape information
  __shared__ sd::LongType inputRank, outputRank;
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
    inputStridePtr = shape::stride(inputShape);
    outputStridePtr = shape::stride(outputShape);

    if (segment < numOfClasses) {
      LongType segmentCoords[] = {segment};
      COORDS2INDEX(1, outputStridePtr, segmentCoords, zIndex);
      start = starts[segment];
      finish = start + lengths[segment];
      LongType startCoords[] = {start};
      LongType xOffset;
      COORDS2INDEX(1, inputStridePtr, startCoords, xOffset);
      z[zIndex] = x[xOffset];
      val[segment] = z[zIndex];
    }
  }
  __syncthreads();

  for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
    LongType eCoords[] = {e};
    LongType xIndex;
    COORDS2INDEX(1, inputStridePtr, eCoords, xIndex);
    math::atomics::sd_atomicMax<T>(&z[zIndex], x[xIndex]);
  }
}

template <typename T, typename I>
static SD_KERNEL void unsortedSegmentMaxLinearKernel(void* input, LongType const* inputShape, void* indices,
                                                     LongType const* indicesShape, LongType* starts,
                                                     LongType* lengths, LongType numOfClasses, void* output,
                                                     LongType const* outputShape) {
  __shared__ LongType xLen, zLen, zIndex;
  __shared__ T* x;
  __shared__ T* z;
  __shared__ I* y;

  // Cache shape information
  __shared__ sd::LongType inputRank, outputRank, indicesRank;
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
    inputStridePtr = shape::stride(inputShape);
    outputStridePtr = shape::stride(outputShape);
    indicesStridePtr = shape::stride(indicesShape);

    LongType segmentCoords[] = {segment};
    COORDS2INDEX(1, outputStridePtr, segmentCoords, zIndex);
    if (lengths[segment] > 0) {
      LongType startCoords[] = {starts[segment]};
      LongType xOffset;
      COORDS2INDEX(1, inputStridePtr, startCoords, xOffset);
      z[zIndex] = x[xOffset];
    } else {
      z[zIndex] = -DataTypeUtils::max<T>();
    }
  }
  __syncthreads();

  if (lengths[segment] > 0) {
    for (auto e = threadIdx.x + 1; e < xLen; e += blockDim.x) {
      LongType eCoords[] = {e};
      LongType xIndex, yIndex;
      COORDS2INDEX(1, inputStridePtr, eCoords, xIndex);
      COORDS2INDEX(1, indicesStridePtr, eCoords, yIndex);
      if (y[yIndex] == segment) {
        math::atomics::sd_atomicMax<T>(&z[zIndex], x[xIndex]);
      }
    }
  }
}

template <typename T, typename I>
static SD_KERNEL void segmentMaxTadKernel(void* inputBuf, LongType const* inputShape, LongType const* inputTads,
                                          LongType const* inputTadOffsets, I* indices, LongType* starts,
                                          LongType* lengths, LongType numOfClasses, void* outputBuf,
                                          LongType const* outputShape, LongType const* outputTads,
                                          LongType const* outputTadOffsets, T filler, LongType indicesLength,
                                          LongType numInputTads, LongType numOutputTads) {
  __shared__ T* val;
  __shared__ LongType len, zIndex, total, zLen;
  __shared__ T* z;
  __shared__ int start, finish;
  __shared__ I segment;

  // Cache shape information
  __shared__ sd::LongType inputTadRank, outputTadRank;
  __shared__ const sd::LongType* inputTadShapePtr;
  __shared__ const sd::LongType* outputTadShapePtr;
  __shared__ const sd::LongType* inputTadStridePtr;
  __shared__ const sd::LongType* outputTadStridePtr;

  if (threadIdx.x == 0 && blockIdx.x < indicesLength) {
    segment = indices[blockIdx.x];
    zLen = shape::length(outputShape);
    auto zOffset = outputTadOffsets[segment];
    z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
    len = shape::length(inputTads);

    // Cache shape information
    inputTadRank = shape::rank(inputTads);
    outputTadRank = shape::rank(outputTads);
    inputTadShapePtr = shape::shapeOf(inputTads);
    outputTadShapePtr = shape::shapeOf(outputTads);
    inputTadStridePtr = shape::stride(inputTads);
    outputTadStridePtr = shape::stride(outputTads);

    start = starts[segment];
    finish = start + lengths[segment];
    total = shape::sizeAt(inputShape, 0);
  }
  __syncthreads();

  auto idx = blockIdx.x;
  if (idx < numInputTads) {
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

        math::atomics::sd_atomicMax<T>(&z[zIndex], x[xIndex]);
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

        if (lengths[segment]) math::atomics::sd_atomicMax<T>(&z[zIndex], x[xIndex]);
      }
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void segmentMaxFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
 T val = -DataTypeUtils::max<T>();
 output->assign(val);
 auto stream = context->getCudaStream();
 indices->syncToHost();
 LongType numOfClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
 NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
 NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
 sd::LongType len = indices->lengthOf();
 classesRangesBegs.assign(len);
 int zero2 = 0;
 classesRangesLens.assign(zero2);
 LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
 LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
 fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);

 NDArray::prepareSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});

 if (input->isVector()  || input->isScalar()) {
   dim3 launchDims = segmentDims(numOfClasses,input->lengthOf());
   segmentMaxLinearKernel<T, I><<<launchDims.y,launchDims.x,launchDims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(),
       output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "segmentMaxLinearKernel failed");

 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   dim3 launchDims = segmentTad(packX->numberOfTads());
   segmentMaxTadKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
       reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
       output->specialShapeInfo(), outputTads, outputTadOffsets,static_cast<T>(0),
       indices->lengthOf(),packX->numberOfTads(),packZ->numberOfTads());
   sd::DebugHelper::checkErrorCode(stream, "segmentMaxTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});

}
// -------------------------------------------------------------------------------------------------------------- //
void segmentMaxFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices});
 BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentMaxFunctor_, (context, input, indices, output),
                       SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices});
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void unsortedSegmentMaxFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
 auto stream = context->getCudaStream();
 T val = DataTypeUtils::infOrMax<T>();
 output->assign(val);

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
   unsortedSegmentMaxLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
   sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentMaxLinearKernel failed");

 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto inputTads = packX->specialShapeInfo();
   auto inputTadOffsets = packX->specialOffsets();
   auto outputTads = packZ->specialShapeInfo();
   auto outputTadOffsets = packZ->specialOffsets();
   dims.x = input->sizeAt(0);
   T val = -DataTypeUtils::max<T>();
   output->assign(val);
   segmentMaxTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
       reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
       output->specialShapeInfo(), outputTads, outputTadOffsets,0,indices->lengthOf(),packX->numberOfTads(),packZ->numberOfTads());

   delete dimensions;
 }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentMaxFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                              NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices});
 output->nullify();
 BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMaxFunctor_,
                       (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
// segment max
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentMaxBPLinearKernel(void* inputBuf, LongType const* inputShape, void* forwardOutput,
                                               LongType const* forwardShape, void* eps, LongType const* epsShape, void* indicesBuf, LongType const* indicesShape, void* outputBuf,
                                               LongType const* outputShape, LongType indicesLen) {
 __shared__ T* x;
 __shared__ T* gradIn;
 __shared__ T* gradOut;
 __shared__ I* y;
 __shared__ T* z;
 __shared__ LongType xLen, gradLen;

 // Cache shape/stride/rank information in shared memory
 __shared__ sd::LongType xRank, yRank, zRank, gradInRank, gradOutRank;
 __shared__ const sd::LongType *xShapePtr, *yShapePtr, *zShapePtr, *gradInShapePtr, *gradOutShapePtr;
 __shared__ const sd::LongType *xStridePtr, *yStridePtr, *zStridePtr, *gradInStridePtr, *gradOutStridePtr;

 if (threadIdx.x == 0) {
   xLen = shape::length(inputShape);
   x = reinterpret_cast<T*>(inputBuf);
   y = reinterpret_cast<I*>(indicesBuf);
   z = reinterpret_cast<T*>(outputBuf);
   gradIn = reinterpret_cast<T*>(forwardOutput);
   gradOut = reinterpret_cast<T*>(eps);
   gradLen = shape::length(epsShape);

   // Cache all shape information
   xRank = shape::rank(inputShape);
   yRank = shape::rank(indicesShape);
   zRank = shape::rank(outputShape);
   gradInRank = shape::rank(forwardShape);
   gradOutRank = shape::rank(epsShape);

   xShapePtr = shape::shapeOf(inputShape);
   yShapePtr = shape::shapeOf(indicesShape);
   zShapePtr = shape::shapeOf(outputShape);
   gradInShapePtr = shape::shapeOf(forwardShape);
   gradOutShapePtr = shape::shapeOf(epsShape);

   xStridePtr = shape::stride(inputShape);
   yStridePtr = shape::stride(indicesShape);
   zStridePtr = shape::stride(outputShape);
   gradInStridePtr = shape::stride(forwardShape);
   gradOutStridePtr = shape::stride(epsShape);
 }
 __syncthreads();

 auto start = blockIdx.x * blockDim.x + threadIdx.x;
 auto step = gridDim.x * blockDim.x;

 for (auto e = start; e < indicesLen; e += step) {
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

   INDEX2COORDS(e, zRank, zShapePtr, zCoords);
   COORDS2INDEX(zRank, zStridePtr, zCoords, zOffset);
   INDEX2COORDS(e, xRank, xShapePtr, xCoords);
   COORDS2INDEX(xRank, xStridePtr, xCoords, xOffset);
   INDEX2COORDS(e, yRank, yShapePtr, yCoords);
   COORDS2INDEX(yRank, yStridePtr, yCoords, yOffset);

   auto classIndex = y[yOffset];
   INDEX2COORDS(classIndex, gradInRank, gradInShapePtr, gradICoords);
   COORDS2INDEX(gradInRank, gradInStridePtr, gradICoords, gradOffsetI);
   INDEX2COORDS(classIndex, gradOutRank, gradOutShapePtr, gradOCoords);
   COORDS2INDEX(gradOutRank, gradOutStridePtr, gradOCoords, gradOffsetO);

   if (math::sd_abs<T,T>(gradIn[gradOffsetI] - x[xOffset]) <= T(1.e-6)) {
     z[zOffset] = gradOut[gradOffsetO];
   }
 }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentMaxBPTadKernel(void* inputBuf, LongType const* inputShape,
                                            void* forwardOutput,
                                            LongType const* forwardShape,
                                            void* eps, LongType const* epsShape,
                                            void* indicesBuf, LongType const* indicesShape,
                                            void* outputBuf,
                                            LongType const* outputShape, LongType const* inputTadShapeInfo,
                                            LongType const* inputOffsets, LongType const* gradInTadShapeInfo,
                                            LongType const* gradInOffsets, LongType const* gradOutTadShapeInfo,
                                            LongType const* gradOutOffsets, LongType const* outTadShapeInfo,
                                            LongType const* outOffsets, LongType indicesLen) {
 __shared__ T* x;
 __shared__ I *indices;
 __shared__ T* gradIn;
 __shared__ T* gradOut;
 __shared__ I* y;
 __shared__ T* z;
 __shared__ LongType xLen, yLen, gradLen, currentLen, gradOutLen, inLen;

 // Cache shape information for all TADs
 __shared__ sd::LongType inputTadRank;
 __shared__ const sd::LongType* inputTadShapePtr;
 __shared__ const sd::LongType* inputTadStridePtr;

 __shared__ sd::LongType gradInTadRank;
 __shared__ const sd::LongType* gradInTadShapePtr;
 __shared__ const sd::LongType* gradInTadStridePtr;

 __shared__ sd::LongType gradOutTadRank;
 __shared__ const sd::LongType* gradOutTadShapePtr;
 __shared__ const sd::LongType* gradOutTadStridePtr;

 __shared__ sd::LongType outTadRank;
 __shared__ const sd::LongType* outTadShapePtr;
 __shared__ const sd::LongType* outTadStridePtr;

 if (threadIdx.x == 0) {
   xLen = shape::length(inputShape);
   indices = reinterpret_cast<I*>(indicesBuf);
   x = reinterpret_cast<T*>(inputBuf);
   y = reinterpret_cast<I*>(indicesBuf);
   z = reinterpret_cast<T*>(outputBuf);
   yLen = shape::length(indicesShape);
   gradOut = reinterpret_cast<T*>(eps);
   gradIn = reinterpret_cast<T*>(forwardOutput);
   gradLen = shape::length(epsShape);
   inLen = shape::length(gradInTadShapeInfo);
   gradOutLen = shape::length(gradOutTadShapeInfo);
   currentLen = shape::length(inputTadShapeInfo);

   // Cache all TAD shape information
   inputTadRank = shape::rank(inputTadShapeInfo);
   inputTadShapePtr = shape::shapeOf(inputTadShapeInfo);
   inputTadStridePtr = shape::stride(inputTadShapeInfo);

   gradInTadRank = shape::rank(gradInTadShapeInfo);
   gradInTadShapePtr = shape::shapeOf(gradInTadShapeInfo);
   gradInTadStridePtr = shape::stride(gradInTadShapeInfo);

   gradOutTadRank = shape::rank(gradOutTadShapeInfo);
   gradOutTadShapePtr = shape::shapeOf(gradOutTadShapeInfo);
   gradOutTadStridePtr = shape::stride(gradOutTadShapeInfo);

   outTadRank = shape::rank(outTadShapeInfo);
   outTadShapePtr = shape::shapeOf(outTadShapeInfo);
   outTadStridePtr = shape::stride(outTadShapeInfo);
 }
 __syncthreads();

 for (auto i = blockIdx.x; i < indicesLen; i += gridDim.x) {
   I segment = indices[i];
   T* current = x;
   T* currentOut = z;
   auto classNum = segment;
   auto currentOffset = inputOffsets[i];
   auto currentOutOffset = outOffsets[i];
   auto currentGradOutOffset = gradOutOffsets[classNum];
   auto bPTensorOffset = gradInOffsets[classNum];

   auto gradIn2 = gradIn + bPTensorOffset;
   auto current2 = current + currentOffset;
   auto currentGradOut2 = gradOut + currentGradOutOffset;
   auto currentOut2 = currentOut + currentOutOffset;

   for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
     sd::LongType xCoords[SD_MAX_RANK];
     sd::LongType gradInCoords[SD_MAX_RANK];
     sd::LongType gradOutCoords[SD_MAX_RANK];
     sd::LongType outCoords[SD_MAX_RANK];
     sd::LongType xIndex;
     sd::LongType gradInIndex;
     sd::LongType gradOutIndex;
     sd::LongType outIndex;

     INDEX2COORDS(e, inputTadRank, inputTadShapePtr, xCoords);
     COORDS2INDEX(inputTadRank, inputTadStridePtr, xCoords, xIndex);
     INDEX2COORDS(e, gradInTadRank, gradInTadShapePtr, gradInCoords);
     COORDS2INDEX(gradInTadRank, gradInTadStridePtr, gradInCoords, gradInIndex);
     INDEX2COORDS(e, gradOutTadRank, gradOutTadShapePtr, gradOutCoords);
     COORDS2INDEX(gradOutTadRank, gradOutTadStridePtr, gradOutCoords, gradOutIndex);
     INDEX2COORDS(e, outTadRank, outTadShapePtr, outCoords);
     COORDS2INDEX(outTadRank, outTadStridePtr, outCoords, outIndex);

     if (math::sd_abs<T, T>(gradIn2[gradInIndex] - current2[xIndex]) <= T(1.e-6)) {
       currentOut2[outIndex] = currentGradOut2[gradOutIndex];
     }
   }
 }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
Status segmentMaxFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                           NDArray* output) {
 // if input is a vector: (as if in doc sample)
 auto stream = context->getCudaStream();
 /*  NDArray tempRes(gradOut->ordering(), gradOut->getShapeAsVector(), DataTypeUtils::fromT<T>(),
                   context); */
 auto outShape = gradOut->getShapeAsVector();
 NDArray tempRes(gradOut->ordering(), outShape, DataTypeUtils::fromT<T>(), context);
 segmentMaxFunctor_<T, I>(context, input, indices, &tempRes);
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
 if (input->isVector()  || input->isScalar()) {
   LongType loop_size = input->lengthOf();
   auto numOfClasses = gradOut->lengthOf();
   dim3 segmentBpDims2 = segmentBpDims(1 + gradOut->lengthOf(),input->lengthOf());
   segmentMaxBPLinearKernel<T, I><<<segmentBpDims2.y, segmentBpDims2.x, segmentBpDims2.z, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
       gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       output->specialBuffer(), output->specialShapeInfo(), indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentMaxBPLinearKernel failed");

 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);

   NDArray::preparePrimaryUse({&tempRes}, {&tempRes});
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto packGradIn = ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
   auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
   LongType const* inputTadShapeInfo = packX->specialShapeInfo();
   LongType const* inputTadOffsets = packX->specialOffsets();
   LongType const* outputTadShapeInfo = packZ->specialShapeInfo();
   LongType const* outputTadOffsets = packZ->specialOffsets();
   LongType const* gradInTadShapeInfo = packGradIn->specialShapeInfo();
   LongType const* gradInTadOffsets = packGradIn->specialOffsets();
   LongType const* gradOutTadShapeInfo = packGradOut->specialShapeInfo();
   LongType const* gradOutTadOffsets = packGradOut->specialOffsets();
   dim3 segmentBpTad2 = segmentBpTad(gradOut->lengthOf(),input->lengthOf());
   segmentMaxBPTadKernel<T, I><<<segmentBpTad2.x, segmentBpTad2.y, segmentBpTad2.z, *stream>>>(
       input->specialBuffer(),
       input->specialShapeInfo(),
       tempRes.specialBuffer(),
       tempRes.specialShapeInfo(),

       gradOut->specialBuffer(),
       gradOut->specialShapeInfo(),

       indices->specialBuffer(),
       indices->specialShapeInfo(),
       output->specialBuffer(),
       output->specialShapeInfo(),
       inputTadShapeInfo,
       inputTadOffsets, gradInTadShapeInfo,
       gradInTadOffsets, gradOutTadShapeInfo,
       gradOutTadOffsets, outputTadShapeInfo,
       outputTadOffsets,
       indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentMaxBPTadKernel failed");

   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
 return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status segmentMaxFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                          NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentMaxFunctorBP_,
                       (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static Status unsortedSegmentMaxFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                          NDArray* gradOut,
                                          LongType numOfClasses, NDArray* output) {
 // if input is a vector: (as if in doc sample)
 auto stream = context->getCudaStream();
 auto outShape = gradOut->getShapeAsVector();
 NDArray tempRes(gradOut->ordering(), outShape, DataTypeUtils::fromT<T>(),
                 context);
 unsortedSegmentMaxFunctor_<T, I>(context, input, indices, numOfClasses, &tempRes);
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
 if (input->isVector()  || input->isScalar()) {
   LongType loop_size = input->lengthOf();
   auto numOfClasses = gradOut->lengthOf();  // indices->e<sd::LongType>(loop_size - 1);
   segmentMaxBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
       gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       output->specialBuffer(), output->specialShapeInfo(),indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentMaxBPLinearKernel failed");
 } else {
   LongType zero = 0;
   std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
   auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
   auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
   auto packGradIn = ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
   auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
   LongType const* inputTads = packX->specialShapeInfo();
   LongType const* inputTadOffsets = packX->specialOffsets();
   LongType const* outputTads = packZ->specialShapeInfo();
   LongType const* outputTadOffsets = packZ->specialOffsets();
   LongType const* gradInTads = packGradIn->specialShapeInfo();
   LongType const* gradInTadOffsets = packGradIn->specialOffsets();
   LongType const* gradOutTads = packGradOut->specialShapeInfo();
   LongType const* gradOutTadOffsets = packGradOut->specialOffsets();


   segmentMaxBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
       input->specialBuffer(),
       input->specialShapeInfo(),
       tempRes.specialBuffer(),
       tempRes.specialShapeInfo(),
       gradOut->specialBuffer(),
       gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
       output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
       gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets, indices->lengthOf());
   sd::DebugHelper::checkErrorCode(stream, "segmentMaxBPTadKernel failed");


   delete dimensions;
 }
 NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
 return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentMaxFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                  LongType numOfClasses, NDArray* output) {
 NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
 BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentMaxFunctorBP_,
                       (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
 NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
