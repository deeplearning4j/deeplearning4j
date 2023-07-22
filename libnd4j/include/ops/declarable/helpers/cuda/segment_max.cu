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
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>
#include <execution/cuda/LaunchDims.h>
namespace sd {
namespace ops {
namespace helpers {

// -------------------------------------------------------------------------------------------------------------- //
// Segment ops linear kernels
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void segmentMaxLinearKernel(void* input, sd::LongType const* inputShape, LongType* starts,
                                             LongType* lengths,
                                             sd::LongType numOfClasses, void* output, sd::LongType const* outputShape) {
  __shared__ T* val;
  __shared__ sd::LongType xLen, zLen, zIndex;
  __shared__ T* x;
  __shared__ T* z;
  __shared__ sd::LongType threadsPerSegment, start, finish;

  auto segment = blockIdx.x;
  if (threadIdx.x == 0) {
    x = reinterpret_cast<T*>(input);
    z = reinterpret_cast<T*>(output);
    extern __shared__ unsigned char shmem[];
    val = reinterpret_cast<T*>(shmem);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    if (segment < numOfClasses) {
      zIndex = shape::getIndexOffset(segment, outputShape);
      if(zIndex >= zLen)
        return;
      start = starts[segment];
      finish = start + lengths[segment];
      auto xOffset = shape::getIndexOffset(start, inputShape);
      if(xOffset >= xLen)
        return;
      z[zIndex] = x[xOffset];
      val[segment] = z[zIndex];
    }
  }
  __syncthreads();

  for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
    auto xIndex = shape::getIndexOffset(e, inputShape);
    if(xIndex >= xLen)
      break;
    sd::math::atomics::sd_atomicMax(&z[zIndex], x[xIndex]);
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void unsortedSegmentMaxLinearKernel(void* input, sd::LongType const* inputShape, void* indices,
                                                     sd::LongType const* indicesShape, LongType* starts,
                                                     LongType* lengths,
                                                     sd::LongType numOfClasses, void* output,
                                                     sd::LongType const* outputShape) {
  __shared__ sd::LongType xLen, zLen, zIndex;
  __shared__ T* x;
  __shared__ T* z;
  __shared__ I* y;
  auto segment = blockIdx.x;

  if (threadIdx.x == 0) {
    x = reinterpret_cast<T*>(input);
    z = reinterpret_cast<T*>(output);
    y = reinterpret_cast<I*>(indices);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    zIndex = shape::getIndexOffset(segment, outputShape);
    if (lengths[segment] > 0)
      z[zIndex] = x[shape::getIndexOffset(starts[segment], inputShape)];
    else
      z[zIndex] = -DataTypeUtils::max<T>();
  }
  __syncthreads();
  if (lengths[segment] > 0)
    for (auto e = threadIdx.x + 1; e < xLen; e += blockDim.x) {
      auto xIndex = shape::getIndexOffset(e, inputShape);
      auto yIndex = shape::getIndexOffset(e, indicesShape);
      if (y[yIndex] == segment) {
        sd::math::atomics::sd_atomicMax(&z[zIndex], x[xIndex]);
      }
    }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentMaxTadKernel(void* inputBuf, sd::LongType const* inputShape, sd::LongType const* inputTads,
                                          sd::LongType const* inputTadOffsets, I* indices, LongType* starts,
                                          LongType* lengths, sd::LongType numOfClasses, void* outputBuf,
                                          sd::LongType const* outputShape, sd::LongType const* outputTads,
                                          sd::LongType const* outputTadOffsets, T filler = 0,
                                          sd::LongType indicesLength = 0) {
  __shared__ T* val;
  __shared__ sd::LongType len, zIndex, total,zLen;
  __shared__ T* z;
  __shared__ int start, finish;
  __shared__ I segment;

  if(blockIdx.x >= indicesLength)
    return;

  if (threadIdx.x == 0 && blockIdx.x < numOfClasses) {
    segment = indices[blockIdx.x];
    zLen = shape::length(outputShape);
    auto zOffset = outputTadOffsets[segment];
    z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
    len = shape::length(inputTads);

    start = starts[segment];
    finish = start + lengths[segment];
    total = shape::sizeAt(inputShape, 0);
  }
  __syncthreads();

  auto idx = blockIdx.x;
  if (idx < total) {
    auto x = reinterpret_cast<T*>(inputBuf) + inputTadOffsets[idx];
    if (blockIdx.x == start) {
      for (auto e = threadIdx.x; e < len; e += blockDim.x) {
        auto xIndex = shape::getIndexOffset(e, inputTads);
        auto zIndex = shape::getIndexOffset(e, outputTads);
        sd::math::atomics::sd_atomicMax(&z[zIndex], x[xIndex]);
      }
    } else {
      for (auto e = threadIdx.x; e < len; e += blockDim.x) {
        auto xIndex = shape::getIndexOffset(e, inputTads);
        auto zIndex = shape::getIndexOffset(e, outputTads);
        if (lengths[segment]) sd::math::atomics::sd_atomicMax(&z[zIndex], x[xIndex]);
      }
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void segmentMaxFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  output->assign(-DataTypeUtils::infOrMax<T>());
  auto stream = context->getCudaStream();
  indices->syncToHost();
  sd::LongType numOfClasses = indices->e<sd::LongType>(indices->lengthOf() - 1) + 1;
  NDArray classesRangesLens = NDArrayFactory::create<sd::LongType>('c', {numOfClasses}, context);
  NDArray classesRangesBegs = NDArrayFactory::create<sd::LongType>('c', {numOfClasses}, context);
  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);
  sd::LongType* begins = reinterpret_cast<sd::LongType*>(classesRangesBegs.specialBuffer());
  sd::LongType* lengths = reinterpret_cast<sd::LongType*>(classesRangesLens.specialBuffer());
  fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);

  NDArray::prepareSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});

  if (input->isVector()) {
    dim3 launchDims = segmentDims(numOfClasses,input->lengthOf());
    segmentMaxLinearKernel<T, I><<<launchDims.y,launchDims.x,launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo());
  } else {
    sd::LongType zero = 0;
    std::vector<sd::LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dim3 launchDims = segmentTad(packX->numberOfTads());
    segmentMaxTadKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});

}
// -------------------------------------------------------------------------------------------------------------- //
void segmentMaxFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentMaxFunctor_, (context, input, indices, output),
                        SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void unsortedSegmentMaxFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                       sd::LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  output->assign(DataTypeUtils::infOrMax<T>());

  NDArray classesRangesBegs = NDArrayFactory::create<sd::LongType>('c', {numOfClasses}, context);
  NDArray classesRangesLens = NDArrayFactory::create<sd::LongType>('c', {numOfClasses}, context);
  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);

  dim3 dims = getFillUpSegmentsDims(numOfClasses, indices->lengthOf());
  fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
  sd::LongType * begins = reinterpret_cast<sd::LongType *>(classesRangesBegs.specialBuffer());
  sd::LongType * lengths = reinterpret_cast<sd::LongType *>(classesRangesLens.specialBuffer());

  if (input->isVector()) {
    unsortedSegmentMaxLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
  } else {
    sd::LongType zero = 0;
    std::vector<sd::LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dims.x = input->sizeAt(0);
    output->assign(-DataTypeUtils::max<T>());
    segmentMaxTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, 0, 0);
    delete dimensions;
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentMaxFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, sd::LongType numOfClasses,
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
static SD_KERNEL void segmentMaxBPLinearKernel(void* inputBuf, sd::LongType const* inputShape, void* forwardOutput,
                                               sd::LongType const* forwardShape, void* eps,
                                               sd::LongType const* epsShape, void* indicesBuf,
                                               sd::LongType const* indicesShape, void* outputBuf,
                                               sd::LongType const* outputShape) {
  __shared__ T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ sd::LongType xLen, gradLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    x = reinterpret_cast<T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradIn = reinterpret_cast<T*>(forwardOutput);
    gradOut = reinterpret_cast<T*>(eps);
    gradLen = shape::length(epsShape);
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;

  for (auto e = start; e < xLen; e += step) {
    auto zOffset = shape::getIndexOffset(e, outputShape);
    auto xOffset = shape::getIndexOffset(e, inputShape);
    auto yOffset = shape::getIndexOffset(e, indicesShape);
    auto classIndex = y[yOffset];
    auto gradOffsetI = shape::getIndexOffset(classIndex, forwardShape);
    auto gradOffsetO = shape::getIndexOffset(classIndex, epsShape);

    if (sd::math::sd_abs(gradIn[gradOffsetI] - x[xOffset]) <= T(1.e-6)) {
      z[zOffset] = gradOut[gradOffsetO];
    }
  }
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentMaxBPTadKernel(void* inputBuf, sd::LongType const* inputShape, void* forwardOutput,
                                            sd::LongType const* forwardShape, void* eps, sd::LongType const* epsShape,
                                            void* indicesBuf, sd::LongType const* indicesShape, void* outputBuf,
                                            sd::LongType const* outputShape, sd::LongType const* inputTad,
                                            sd::LongType const* inputOffsets, sd::LongType const* gradInTad,
                                            sd::LongType const* gradInOffsets, sd::LongType const* gradOutTad,
                                            sd::LongType const* gradOutOffsets, sd::LongType const* outTad,
                                            sd::LongType const* outOffsets) {
  __shared__ T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ sd::LongType xLen, yLen, gradLen, currentLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    x = reinterpret_cast<T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    yLen = shape::length(indicesShape);
    gradOut = reinterpret_cast<T*>(eps);
    gradIn = reinterpret_cast<T*>(forwardOutput);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outTad);
  }
  __syncthreads();

  for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
    auto yIndex = shape::getIndexOffset(i, indicesShape);
    auto segment = y[yIndex];
    T* current = x + inputOffsets[i];
    T* currentOut = z + outOffsets[i];
    T* in = gradIn + gradInOffsets[segment];
    T* outGrad = gradOut + gradOutOffsets[segment];

    for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
      if (sd::math::sd_abs(in[e] - current[e]) <= T(1.e-6)) currentOut[e] = outGrad[e];
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
sd::Status segmentMaxFunctorBP_(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                NDArray* output) {
  // if input is a vector: (as if in doc sample)
  auto stream = context->getCudaStream();
  NDArray tempRes(gradOut->ordering(), gradOut->getShapeAsVector(), DataTypeUtils::fromT<T>(),
                  context);  //->shapeInfo(), context);
  segmentMaxFunctor_<T, I>(context, input, indices, &tempRes);
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
  if (input->isVector()) {
    sd::LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();
    dim3 segmentBpDims2 = segmentBpDims(1 + gradOut->lengthOf(),input->lengthOf());

    segmentMaxBPLinearKernel<T, I><<<segmentBpDims2.y,segmentBpDims2.x, segmentBpDims2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo());
  } else {
    sd::LongType zero = 0;
    std::vector<sd::LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto packGradIn = sd::ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
    auto packGradOut = sd::ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    sd::LongType const* inputTads = packX->specialShapeInfo();
    sd::LongType const* inputTadOffsets = packX->specialOffsets();
    sd::LongType const* outputTads = packZ->specialShapeInfo();
    sd::LongType const* outputTadOffsets = packZ->specialOffsets();
    sd::LongType const* gradInTads = packGradIn->specialShapeInfo();
    sd::LongType const* gradInTadOffsets = packGradIn->specialOffsets();
    sd::LongType const* gradOutTads = packGradOut->specialShapeInfo();
    sd::LongType const* gradOutTadOffsets = packGradOut->specialOffsets();
    dim3 segmentBpTad2 = segmentBpTad(gradOut->lengthOf(),input->lengthOf());
    segmentMaxBPTadKernel<T, I><<<segmentBpTad2.y, segmentBpTad2.x, segmentBpTad2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
        gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
  return sd::Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
sd::Status segmentMaxFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                               NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentMaxFunctorBP_,
                        (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static sd::Status unsortedSegmentMaxFunctorBP_(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                               NDArray* gradOut, sd::LongType numOfClasses, NDArray* output) {
  // if input is a vector: (as if in doc sample)
  auto stream = context->getCudaStream();
  NDArray tempRes(gradOut->ordering(), gradOut->getShapeAsVector(), DataTypeUtils::fromT<T>(),
                  context);  //->shapeInfo(), context);
  unsortedSegmentMaxFunctor_<T, I>(context, input, indices, numOfClasses, &tempRes);
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
  if (input->isVector()) {
    sd::LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();  // indices->e<sd::LongType>(loop_size - 1);
    segmentMaxBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo());
  } else {
    sd::LongType zero = 0;
    std::vector<sd::LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto packGradIn = sd::ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
    auto packGradOut = sd::ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    sd::LongType const* inputTads = packX->specialShapeInfo();
    sd::LongType const* inputTadOffsets = packX->specialOffsets();
    sd::LongType const* outputTads = packZ->specialShapeInfo();
    sd::LongType const* outputTadOffsets = packZ->specialOffsets();
    sd::LongType const* gradInTads = packGradIn->specialShapeInfo();
    sd::LongType const* gradInTadOffsets = packGradIn->specialOffsets();
    sd::LongType const* gradOutTads = packGradOut->specialShapeInfo();
    sd::LongType const* gradOutTadOffsets = packGradOut->specialOffsets();

    segmentMaxBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
        gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
  return sd::Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
sd::Status unsortedSegmentMaxFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                       sd::LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentMaxFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
