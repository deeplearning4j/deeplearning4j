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
    template <typename T, typename I>
    static __global__ void unsortedSegmentSqrtNLinearKernel(T* input, Nd4jLong* inputShape, I* indices, Nd4jLong* indicesShape, int* starts, int* lengths, Nd4jLong numOfClasses, T* output, Nd4jLong* outputShape) {
        __shared__ Nd4jLong xLen, zLen;

        if (threadIdx.x == 0) {
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);
        }
        __syncthreads();

        auto start = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

        for  (auto idx = start; idx < xLen; idx += step) {
            auto yIndex = shape::getIndexOffset(idx, indicesShape);
            auto segment = indices[yIndex];
            auto zIndex = shape::getIndexOffset(segment, outputShape);
            if (lengths[segment] == 0) continue;
            auto xIndex = shape::getIndexOffset(idx, inputShape);

            sd::math::atomics::nd4j_atomicAdd(&output[zIndex],  input[xIndex] / sd::math::nd4j_sqrt<int, T>(lengths[segment]));
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    // SegmentSqrtN kernel
    template <typename T, typename I>
    static __global__ void segmentSqrtNTadKernel(T* inputBuf, Nd4jLong* inputShape, Nd4jLong* inputTads, Nd4jLong* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf, Nd4jLong* outputShape, Nd4jLong* outputTads, Nd4jLong* outputTadOffsets) {

        __shared__ Nd4jLong len, total;

        if (threadIdx.x == 0) {
            total = shape::sizeAt(inputShape, 0);
            len = shape::length(inputTads);
        }
        __syncthreads();

        for (auto idx = blockIdx.x; idx  < total; idx += gridDim.x) {
            auto segment = indices[idx];
            auto x = inputBuf + inputTadOffsets[idx];
            auto z = reinterpret_cast<T *>(outputBuf) + outputTadOffsets[segment];
            auto start = starts[segment];
            auto finish = start + lengths[segment];

            for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputTads);
                auto zIndex = shape::getIndexOffset(e, outputTads);
                sd::math::atomics::nd4j_atomicAdd(&z[zIndex], x[xIndex] / sd::math::nd4j_sqrt<int, T>(lengths[segment]));
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static void unsortedSegmentSqrtNFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses});
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses});
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(sd::BroadcastOpsTuple::Assign(), &row, &classes);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
//        dim3 dims(numOfClasses, indices->lengthOf(), numOfClasses * 32 + 32);
        dim3 dims(128, 256, 256);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
        fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        output->nullify();
        if (input->isVector()) {
            unsortedSegmentSqrtNLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(
                    input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(),
                    indices->dataBuffer()->specialAsT<I>(), indices->specialShapeInfo(), begins, lengths, numOfClasses,
                    output->dataBuffer()->specialAsT<T>(), output->specialShapeInfo());
        }
        else {
            output->nullify();
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            segmentSqrtNTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(
                    input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(), inputTads, inputTadOffsets, indices->dataBuffer()->specialAsT<I>(),
                    begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads,  outputTadOffsets);
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    void unsortedSegmentSqrtNFunctor(sd::LaunchContext* context , NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices});
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentSqrtNFunctor_, (context, input, indices, numOfClasses, output),
                              FLOAT_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices});
    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void segmentSqrtNBPLinearKernel(void* inputBuf, Nd4jLong* inputShape, void* eps, Nd4jLong* epsShape, void* indicesBuf, Nd4jLong* indicesShape,
                                                      int* lengths, void* outputBuf, Nd4jLong* outputShape) {
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
            auto gradOffsetO = shape::getIndexOffset(classIndex, epsShape);

            z[zOffset] = T(gradOut[gradOffsetO] / math::nd4j_sqrt<int, float>(lengths[classIndex]));
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void segmentSqrtNBPTadKernel(void* inputBuf, Nd4jLong* inputShape, void* eps, Nd4jLong* epsShape,
                                                   void* indicesBuf, Nd4jLong* indicesShape, int* lengths, void* outputBuf, Nd4jLong* outputShape,Nd4jLong* inputTad,
                                                   Nd4jLong* inputOffsets, Nd4jLong* gradOutTad, Nd4jLong* gradOutOffsets, Nd4jLong* outTad, Nd4jLong* outOffsets) {
        __shared__ T* x;
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
            gradLen = shape::length(epsShape);
            currentLen = shape::length(outTad);
        }
        __syncthreads();

        for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
//            auto yIndex = shape::getIndexOffset(i, indicesShape);
            auto segment = y[i]; //yIndex];
            T* currentOut = z + outOffsets[i];
            T* outGrad = gradOut + gradOutOffsets[segment];

            for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
                auto zIndex = shape::getIndexOffset(e, outTad);
                auto gradIndex = shape::getIndexOffset(e, gradOutTad);
                if (lengths[segment] > 0)
                    currentOut[zIndex] = T(outGrad[gradIndex] / math::nd4j_sqrt<int, float>(lengths[segment]));
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static int unsortedSegmentSqrtNFunctorBP_(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        auto numClasses = indices->e<int>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses});

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());

        if (input->isVector()) {
            Nd4jLong loop_size = input->lengthOf();
            auto numOfClasses = gradOut->lengthOf(); //indices->e<Nd4jLong>(loop_size - 1);
            segmentSqrtNBPLinearKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(),
                    input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
//            auto packGradIn = sd::ConstantTadHelper::getInstance()->tadForDimensions(tempRes.getShapeInfo(), dimensions);
            auto packGradOut = sd::ConstantTadHelper::getInstance()->tadForDimensions(gradOut->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            Nd4jLong* gradOutTads = packGradOut.specialShapeInfo();
            Nd4jLong* gradOutTadOffsets = packGradOut.specialOffsets();

            segmentSqrtNBPTadKernel<T,I><<<indices->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                    gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), lengths,
                    output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets,
                    outputTads, outputTadOffsets);
        }
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});

        return Status::OK();
    }
    // -------------------------------------------------------------------------------------------------------------- //
    int unsortedSegmentSqrtNFunctorBP(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentSqrtNFunctorBP_, (context, input, indices, gradOut, numOfClasses, output), FLOAT_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
    }
}
}
}