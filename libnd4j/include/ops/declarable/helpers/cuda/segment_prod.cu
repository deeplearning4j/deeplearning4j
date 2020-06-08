/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>
#include <array/NDArrayFactory.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace sd {
namespace ops {
namespace helpers {
    // -------------------------------------------------------------------------------------------------------------- //
    // Segment Prod ops linear kernels
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void segmentProdLinearKernel(void* input, Nd4jLong const* inputShape, int* starts, int* lengths,
            Nd4jLong numOfClasses, void* output, Nd4jLong const* outputShape) {

        __shared__ Nd4jLong xLen, zLen;
        __shared__ T* x;
        __shared__ T* z;

        if (threadIdx.x == 0) {
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);
        }
        __syncthreads();

        for(auto segment = blockIdx.x; segment < numOfClasses; segment += gridDim.x) {
            auto zIndex = shape::getIndexOffset(segment, outputShape);
            auto start = starts[segment];
            auto finish = start + lengths[segment];
            if (lengths[segment] == 0) {
                continue;
            }
            for (auto e = start + threadIdx.x; e < finish; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputShape);
                sd::math::atomics::nd4j_atomicMul(&z[segment], x[xIndex]);
            }
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void unsortedSegmentProdLinearKernel(T* input, Nd4jLong const* inputShape, I* indices, Nd4jLong const* indicesShape, int* starts, int* lengths, Nd4jLong numOfClasses, T* output, Nd4jLong const* outputShape) {
        __shared__ Nd4jLong xLen, zLen;

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
            sd::math::atomics::nd4j_atomicMul(&output[zIndex], input[xIndex]);
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    // SegmentProd kernel
    template <typename T, typename I>
    static __global__ void segmentProdTadKernel(void* inputBuf, Nd4jLong const* inputShape, Nd4jLong const* inputTads,
            Nd4jLong const* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf,
            Nd4jLong const* outputShape, Nd4jLong const* outputTads, Nd4jLong const* outputTadOffsets) {

        __shared__ Nd4jLong len, total;

        if (threadIdx.x == 0) {
            total = shape::sizeAt(inputShape, 0);
            len = shape::length(inputTads);
        }
        __syncthreads();

        for (auto idx = blockIdx.x; idx < total; idx += gridDim.x) {
            auto x = reinterpret_cast<T *>(inputBuf) + inputTadOffsets[idx];
            auto segment = indices[idx]; // / threadsPerSegment;
            auto z = reinterpret_cast<T *>(outputBuf) + outputTadOffsets[segment];
            auto start = starts[segment];
            auto finish = start + lengths[segment];
            if (lengths[segment] == 0) continue;
            for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputTads);
                auto zIndex = shape::getIndexOffset(e, outputTads);
                sd::math::atomics::nd4j_atomicMul(&z[zIndex], x[xIndex]);
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static void segmentProdFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
        auto stream = context->getCudaStream();
        Nd4jLong numClasses = indices->e<Nd4jLong>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses}, context);
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses}, context);
        output->assign(1);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);

        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());

        if (input->isVector()) {
            segmentProdLinearKernel<T,I><<<128, 256, 128, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
            auto inputTads = packX.specialShapeInfo();
            auto inputTadOffsets = packX.specialOffsets();
            auto outputTads = packZ.specialShapeInfo();
            auto outputTadOffsets = packZ.specialOffsets();
            segmentProdTadKernel<T,I><<<128, 512, 2048, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //
    void segmentProdFunctor(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices});
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segmentProdFunctor_, (context, input, indices, output), NUMERIC_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices});
    }

    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static void unsortedSegmentProdFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses}, context);
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses}, context);
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(sd::BroadcastOpsTuple::Assign(), &row, &classes);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numOfClasses, indices->lengthOf(), numOfClasses * 32 + 32);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
        fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        output->assign(1);

        if (input->isVector()) {
            unsortedSegmentProdLinearKernel<T,I><<<128, 256, 256, *stream>>>(
                    input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(),
                    indices->dataBuffer()->specialAsT<I>(), indices->specialShapeInfo(), begins, lengths, numOfClasses,
                    output->dataBuffer()->specialAsT<T>(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
            auto inputTads = packX.specialShapeInfo();
            auto inputTadOffsets = packX.specialOffsets();
            auto outputTads = packZ.specialShapeInfo();
            auto outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            segmentProdTadKernel<T,I><<<128, 256, 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //
    void unsortedSegmentProdFunctor(sd::LaunchContext* context , NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices});
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentProdFunctor_, (context, input, indices, numOfClasses, output),
                              NUMERIC_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices});
    }

    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void segmentProdBPLinearKernel(void* inputBuf, Nd4jLong const* inputShape, void* forwardOutput,
                                                     Nd4jLong const* forwardShape, void* eps, Nd4jLong const* epsShape, void* indicesBuf, Nd4jLong const* indicesShape,
                                                     void* outputBuf, Nd4jLong const* outputShape) {
        __shared__ T* x;
        __shared__ T* gradIn;
        __shared__ T* gradOut;
        __shared__ I* y;
        __shared__ T* z;
        __shared__ Nd4jLong xLen, gradLen;

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

            z[zOffset] = gradOut[gradOffsetO]  * gradIn[gradOffsetI] / x[xOffset];
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void segmentProdBPTadKernel(void* inputBuf, Nd4jLong const* inputShape, void* forwardOutput,
                                                  Nd4jLong const* forwardShape, void* eps, Nd4jLong const* epsShape, void* indicesBuf, Nd4jLong const* indicesShape,
                                                  void* outputBuf, Nd4jLong const* outputShape, Nd4jLong const* inputTad,
                                                  Nd4jLong const* inputOffsets, Nd4jLong const* gradInTad, Nd4jLong const* gradInOffsets,
                                                  Nd4jLong const* gradOutTad, Nd4jLong const* gradOutOffsets, Nd4jLong const* outTad,
                                                  Nd4jLong const* outOffsets) {
        __shared__ T* x;
        __shared__ T* gradIn;
        __shared__ T* gradOut;
        __shared__ I* y;
        __shared__ T* z;
        __shared__ Nd4jLong xLen, yLen, gradLen, currentLen;

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
    int segmentProdFunctorBP_(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray tempRes(gradOut->ordering(), gradOut->getShapeAsVector(), DataTypeUtils::fromT<T>(), context);//->shapeInfo(), context);
        segmentProdFunctor_<T, I>(context, input, indices, &tempRes);
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        if (input->isVector()) {
            Nd4jLong loopSize = input->lengthOf();
            auto numOfClasses = gradOut->lengthOf(); //indices->e<Nd4jLong>(loop_size - 1);
            segmentProdBPLinearKernel<T,I><<<gradOut->lengthOf(), loopSize, 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                    tempRes.specialBuffer(), tempRes.specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
            auto packGradIn = sd::ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
            auto packGradOut = sd::ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
            auto inputTads = packX.specialShapeInfo();
            auto inputTadOffsets = packX.specialOffsets();
            auto outputTads = packZ.specialShapeInfo();
            auto outputTadOffsets = packZ.specialOffsets();
            auto gradInTads = packGradIn.specialShapeInfo();
            auto gradInTadOffsets = packGradIn.specialOffsets();
            auto gradOutTads = packGradOut.specialShapeInfo();
            auto gradOutTadOffsets = packGradOut.specialOffsets();

            segmentProdBPTadKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                    tempRes.specialBuffer(), tempRes.specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
                    inputTads, inputTadOffsets, gradInTads, gradInTadOffsets, gradOutTads, gradOutTadOffsets,
                    outputTads, outputTadOffsets);
        }
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
        return Status::OK();
    }

    // -------------------------------------------------------------------------------------------------------------- //

    int segmentProdFunctorBP(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentProdFunctorBP_, (context, input,
                indices, gradOut, output), FLOAT_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
    }

    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static int unsortedSegmentProdFunctorBP_(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();

        NDArray tempRes(gradOut->ordering(), gradOut->getShapeAsVector(), DataTypeUtils::fromT<T>(), context);//->shapeInfo(), context);
        unsortedSegmentProdFunctor_<T, I>(context, input, indices, numOfClasses, &tempRes);
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        if (input->isVector()) {
            Nd4jLong loopSize = input->lengthOf();
            auto numOfClasses = gradOut->lengthOf(); //indices->e<Nd4jLong>(loop_size - 1);
            segmentProdBPLinearKernel<T,I><<<gradOut->lengthOf(), loopSize, 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                    tempRes.specialBuffer(), tempRes.specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
            auto packGradIn = sd::ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
            auto packGradOut = sd::ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
            auto inputTads = packX.specialShapeInfo();
            auto inputTadOffsets = packX.specialOffsets();
            auto outputTads = packZ.specialShapeInfo();
            auto outputTadOffsets = packZ.specialOffsets();
            auto gradInTads = packGradIn.specialShapeInfo();
            auto gradInTadOffsets = packGradIn.specialOffsets();
            auto gradOutTads = packGradOut.specialShapeInfo();
            auto gradOutTadOffsets = packGradOut.specialOffsets();

            segmentProdBPTadKernel<T,I><<<indices->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                    tempRes.specialBuffer(), tempRes.specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
                    inputTads, inputTadOffsets, gradInTads, gradInTadOffsets, gradOutTads, gradOutTadOffsets,
                    outputTads, outputTadOffsets);
        }
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
        return Status::OK();
    }

    // -------------------------------------------------------------------------------------------------------------- //
    int unsortedSegmentProdFunctorBP(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentProdFunctorBP_, (context, input, indices, gradOut, numOfClasses, output), FLOAT_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
    }

    // -------------------------------------------------------------------------------------------------------------- //

}
}
}
