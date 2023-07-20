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
// Segment Prod ops linear kernels
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void segmentProdLinearKernel(void* input, sd::LongType const* inputShape, sd::LongType* starts, sd::LongType* lengths,
                                              sd::LongType numOfClasses, void* output,
                                              sd::LongType const* outputShape) {
  __shared__ sd::LongType xLen, zLen;
  __shared__ T* x;
  __shared__ T* z;

  if (threadIdx.x == 0) {
    x = reinterpret_cast<T*>(input);
    z = reinterpret_cast<T*>(output);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);
  }
  __syncthreads();

  for (auto segment = blockIdx.x; segment < numOfClasses; segment += gridDim.x) {
    auto zIndex = shape::getIndexOffset(segment, outputShape);
    if(zIndex >= zLen)
      continue;
    auto start = starts[segment];
    auto finish = start + lengths[segment];
    if (lengths[segment] == 0) {
      continue;
    }
    for (auto e = start + threadIdx.x; e < finish; e += blockDim.x) {
      auto xIndex = shape::getIndexOffset(e, inputShape);
      if(xIndex >= xLen)
        return;
      sd::math::atomics::sd_atomicMul(&z[segment], x[xIndex]);
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void unsortedSegmentProdLinearKernel(T* input, sd::LongType const* inputShape, I* indices,
                                                      sd::LongType const* indicesShape, sd::LongType* starts, sd::LongType* lengths,
                                                      sd::LongType numOfClasses, T* output,
                                                      sd::LongType const* outputShape) {
  __shared__ sd::LongType xLen, zLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);
  }
  __syncthreads();
  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;
  for (auto idx = start; idx < xLen; idx += step) {
    auto xIndex = shape::getIndexOffset(idx, inputShape);
    auto yIndex = shape::getIndexOffset(idx, indicesShape);
    auto segment = indices[yIndex];
    auto zIndex = shape::getIndexOffset(segment, outputShape);
    if (lengths[segment] == 0) {
      continue;
    }
    sd::math::atomics::sd_atomicMul(&output[zIndex], input[xIndex]);
  }
}
// -------------------------------------------------------------------------------------------------------------- //
// SegmentProd kernel
template <typename T, typename I>
static SD_KERNEL void segmentProdTadKernel(void* inputBuf, sd::LongType const* inputShape,
                                           sd::LongType const* inputTads, sd::LongType const* inputTadOffsets,
                                           I* indices, sd::LongType* starts, sd::LongType* lengths,
                                           sd::LongType numOfClasses, void* outputBuf, sd::LongType const* outputShape,
                                           sd::LongType const* outputTads, sd::LongType const* outputTadOffsets) {
  __shared__ sd::LongType len, total;

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
      auto xIndex = shape::getIndexOffset(e, inputTads);
      auto zIndex = shape::getIndexOffset(e, outputTads);
      sd::math::atomics::sd_atomicMul(&z[zIndex], x[xIndex]);
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void segmentProdFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  auto stream = context->getCudaStream();
  sd::LongType numClasses = indices->e<sd::LongType>(indices->lengthOf() - 1) + 1;
  NDArray classesRangesLens = NDArrayFactory::create<sd::LongType>('c', {numClasses}, context);
  NDArray classesRangesBegs = NDArrayFactory::create<sd::LongType>('c', {numClasses}, context);
  output->assign(1);
  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);

  fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
  sd::LongType* begins = reinterpret_cast<sd::LongType*>(classesRangesBegs.specialBuffer());
  sd::LongType* lengths = reinterpret_cast<sd::LongType*>(classesRangesLens.specialBuffer());

  if (input->isVector()) {
    sd_print("Input is vector segment prod\n");
    dim3 launchDims = getLaunchDims("segment_prod_2");
    segmentProdLinearKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins,
                                                                                         lengths, numClasses, output->specialBuffer(),
                                                                                         output->specialShapeInfo());
  } else {
    sd_print("Other case segment prod\n");
    sd::LongType zero = 0;
    std::vector<sd::LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dim3 launchDims = getLaunchDims("segment_prod_2_tad");
    segmentProdTadKernel<T, I><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets);
    delete dimensions;
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void segmentProdFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segmentProdFunctor_, (context, input, indices, output),
                        SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static void unsortedSegmentProdFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                        sd::LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray classesRangesBegs = NDArrayFactory::create<sd::LongType>('c', {numOfClasses}, context);
  NDArray classesRangesLens = NDArrayFactory::create<sd::LongType>('c', {numOfClasses}, context);
  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);
  dim3 dims = getFillUpSegmentsDims(numOfClasses,indices->lengthOf());
  fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
  sd::LongType* begins = reinterpret_cast<sd::LongType*>(classesRangesBegs.specialBuffer());
  sd::LongType* lengths = reinterpret_cast<sd::LongType*>(classesRangesLens.specialBuffer());
  output->assign(1);

  dim3 launchDims = getLaunchDims("unsorted_segment_prod_2");
  if (input->isVector()) {
    unsortedSegmentProdLinearKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(), indices->dataBuffer()->specialAsT<I>(),
        indices->specialShapeInfo(), begins, lengths, numOfClasses, output->dataBuffer()->specialAsT<T>(),
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
    dims.x = input->sizeAt(0);
    segmentProdTadKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets);
    delete dimensions;
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentProdFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, sd::LongType numOfClasses,
                                NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentProdFunctor_,
                        (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentProdBPLinearKernel(void* inputBuf, sd::LongType const* inputShape, void* forwardOutput,
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

    z[zOffset] = gradOut[gradOffsetO] * gradIn[gradOffsetI] / x[xOffset];
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentProdBPTadKernel(void* inputBuf, sd::LongType const* inputShape, void* forwardOutput,
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
      currentOut[e] = outGrad[e] * in[e] / current[e];
    }
  }
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
sd::Status segmentProdFunctorBP_(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                 NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray tempRes(gradOut->ordering(), gradOut->getShapeAsVector(), DataTypeUtils::fromT<T>(),
                  context);  //->shapeInfo(), context);
  segmentProdFunctor_<T, I>(context, input, indices, &tempRes);
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  if (input->isVector()) {
    sd::LongType loopSize = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();  // indices->e<sd::LongType>(loop_size - 1);
    segmentProdBPLinearKernel<T, I><<<gradOut->lengthOf(), loopSize, 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo());
  } else {
    sd::LongType zero = 0;
    std::vector<sd::LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,&zero);
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto packGradIn = sd::ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
    auto packGradOut = sd::ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    auto gradInTads = packGradIn->specialShapeInfo();
    auto gradInTadOffsets = packGradIn->specialOffsets();
    auto gradOutTads = packGradOut->specialShapeInfo();
    auto gradOutTadOffsets = packGradOut->specialOffsets();
    dim3 segmentBpTad2 = segmentBpTad(gradOut->lengthOf(),input->lengthOf());

    segmentProdBPTadKernel<T, I><<<segmentBpTad2.y,segmentBpTad2.x, segmentBpTad2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
        gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  return sd::Status::OK;
}

// -------------------------------------------------------------------------------------------------------------- //

sd::Status segmentProdFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentProdFunctorBP_,
                        (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static sd::Status unsortedSegmentProdFunctorBP_(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                                NDArray* gradOut, sd::LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();

  NDArray tempRes(gradOut->ordering(), gradOut->getShapeAsVector(), DataTypeUtils::fromT<T>(),
                  context);
  unsortedSegmentProdFunctor_<T, I>(context, input, indices, numOfClasses, &tempRes);
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  if (input->isVector()) {
    sd::LongType loopSize = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();
    dim3 segmentBpTad2 = segmentBpDims(gradOut->lengthOf(),input->lengthOf());
    segmentProdBPLinearKernel<T, I><<<segmentBpTad2.y, segmentBpTad2.x,segmentBpTad2.z, *stream>>>(
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
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    auto gradInTads = packGradIn->specialShapeInfo();
    auto gradInTadOffsets = packGradIn->specialOffsets();
    auto gradOutTads = packGradOut->specialShapeInfo();
    auto gradOutTadOffsets = packGradOut->specialOffsets();
    dim3 segmentBpTad2 = segmentBpTad(gradOut->lengthOf(),input->lengthOf());
    segmentProdBPTadKernel<T, I><<<indices->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
        gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  return sd::Status::OK;
}

// -------------------------------------------------------------------------------------------------------------- //
sd::Status unsortedSegmentProdFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                        sd::LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentProdFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

// -------------------------------------------------------------------------------------------------------------- //

}  // namespace helpers
}  // namespace ops
}  // namespace sd
