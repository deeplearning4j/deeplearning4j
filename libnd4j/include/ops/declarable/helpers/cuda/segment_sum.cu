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
#include <NDArrayFactory.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>

namespace nd4j {
namespace ops {
namespace helpers {
    // -------------------------------------------------------------------------------------------------------------- //
    // Segment ops linear kernels
    // -------------------------------------------------------------------------------------------------------------- //
    template<typename T, typename I>
    static __global__ void
    segmentSumLinearKernel(void *input, Nd4jLong *inputShape, int *starts, int *lengths, Nd4jLong numOfClasses,
                           void *output, Nd4jLong *outputShape) {
        __shared__
        T *val;
        __shared__
        Nd4jLong xLen, zLen, segment, zIndex;
        __shared__
        T *x;
        __shared__
        T *z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x / threadsPerSegment;
            x = reinterpret_cast<T *>(input);
            z = reinterpret_cast<T *>(output);

            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);


            if (segment < numOfClasses) {
                zIndex = shape::getIndexOffset(segment, outputShape, zLen);
                start = starts[segment];
                finish = start + lengths[segment];
                //val[segment] = ;
                z[zIndex] = x[shape::getIndexOffset(start, inputShape, xLen)];
            }

        }
        __syncthreads();

        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
            nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], x[xIndex]);
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template<typename T, typename I>
    static __global__ void
    unsortedSegmentSumLinearKernel(void *input, Nd4jLong *inputShape, void *indices, Nd4jLong *indicesShape,
                                   int *starts, int *lengths, Nd4jLong numOfClasses, void *output,
                                   Nd4jLong *outputShape) {
        __shared__
        T *val;
        __shared__
        Nd4jLong xLen, zLen, segment, zIndex;
        __shared__
        T *x;
        __shared__
        T *z;
        __shared__
        I *y; //int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = blockIdx.x;
            x = reinterpret_cast<T *>(input);
            z = reinterpret_cast<T *>(output);
            y = reinterpret_cast<I *>(indices);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            zIndex = shape::getIndexOffset(segment, outputShape, zLen);
            if (lengths[segment] > 0)
                z[zIndex] = x[shape::getIndexOffset(starts[segment], inputShape, xLen)];
            else
                z[zIndex] = 0; //DataTypeUtils::max<T>();
        }
        __syncthreads();

        if (lengths[segment] > 0)
            for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
                auto yIndex = shape::getIndexOffset(e, indicesShape, xLen);
                if (y[yIndex] == segment && e != starts[segment]) {
                    nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], x[xIndex]);
                }
            }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    // SegmentSum kernel
    template <typename T, typename I>
    static __global__ void segmentSumTadKernel(void* inputBuf, Nd4jLong* inputShape, Nd4jLong* inputTads, Nd4jLong* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf, Nd4jLong* outputShape, Nd4jLong* outputTads, Nd4jLong* outputTadOffsets) {
        __shared__ T* val;
        __shared__ Nd4jLong len, segment, zIndex, total;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = indices[blockIdx.x]; // / threadsPerSegment;
            z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
            len = shape::length(inputTads);
            start = starts[segment];
            finish = start + lengths[segment];
            total = shape::sizeAt(inputShape, 0);

        }
        __syncthreads();

        auto idx = blockIdx.x;
        if (blockIdx.x <= total) {
            auto x = reinterpret_cast<T *>(inputBuf) + inputTadOffsets[idx];
            if (blockIdx.x == start) {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    z[zIndex] = x[xIndex];
                }
            }
            else {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    if (lengths[segment])
                        nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], x[xIndex]);
                }
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static void segmentSumFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
        auto stream = context->getCudaStream();
        Nd4jLong numClasses = indices->e<Nd4jLong>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses});

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);

        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());

        if (input->isVector()) {
            segmentSumLinearKernel<T,I><<<numClasses, input->lengthOf(), numClasses * 32 + 32, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            segmentSumTadKernel<T,I><<<input->sizeAt(0), 512, 2048, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //
    void segmentSumFunctor(nd4j::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentSumFunctor_, (context, input, indices, output), NUMERIC_TYPES, INDEXING_TYPES);
    }

    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static void unsortedSegmentSumFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses});
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses});
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(nd4j::BroadcastOpsTuple::Assign(), &row, &classes);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numOfClasses, indices->lengthOf(), (numOfClasses + 1) * 64);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
        fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());

        if (input->isVector()) {
            unsortedSegmentSumLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            output->assign(0);
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            segmentSumTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //
    void unsortedSegmentSumFunctor(nd4j::LaunchContext* context , NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentSumFunctor_, (context, input, indices, numOfClasses, output),
                              NUMERIC_TYPES, INDEXING_TYPES);

    }

    // -------------------------------------------------------------------------------------------------------------- //
    // Backpropagate ops
    // -------------------------------------------------------------------------------------------------------------- //
    // Sorted sum backpropagate
    template <typename T, typename I>
    static __global__ void segmentSumBPLinearKernel(void* inputBuf, Nd4jLong* inputShape, void* eps, Nd4jLong* epsShape,
                                                    void* indicesBuf, Nd4jLong* indicesShape, void* outputBuf, Nd4jLong* outputShape) {
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

            auto zOffset = shape::getIndexOffset(e, outputShape, xLen);
            auto xOffset = shape::getIndexOffset(e, inputShape, xLen);
            auto yOffset = shape::getIndexOffset(e, indicesShape, xLen);
            auto classIndex = y[yOffset];
            auto gradOffsetO = shape::getIndexOffset(classIndex, epsShape, gradLen);

            z[zOffset] = gradOut[gradOffsetO];
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void segmentSumBPTadKernel(void* inputBuf, Nd4jLong* inputShape, void* eps, Nd4jLong* epsShape,
                                                 void* indicesBuf, Nd4jLong* indicesShape, void* outputBuf, Nd4jLong* outputShape, Nd4jLong* inputTad,
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
            auto yIndex = shape::getIndexOffset(i, indicesShape, yLen);
            auto segment = y[yIndex];
            T* currentOut = z + outOffsets[i];
            T* outGrad = gradOut + gradOutOffsets[segment];

            for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
                currentOut[e] = outGrad[e];
            }
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    int segmentSumFunctorBP_(nd4j::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        if (input->isVector()) {
            Nd4jLong loop_size = input->lengthOf();
            auto numOfClasses = gradOut->lengthOf(); //indices->e<Nd4jLong>(loop_size - 1);
            segmentSumBPLinearKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(),
                    input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            auto packGradOut = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(gradOut->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            Nd4jLong* gradOutTads = packGradOut.specialShapeInfo();
            Nd4jLong* gradOutTadOffsets = packGradOut.specialOffsets();

            segmentSumBPTadKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                    gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
                    inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets,
                    outputTads, outputTadOffsets);
        }
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
        return Status::OK();
    }
    // -------------------------------------------------------------------------------------------------------------- //

    int segmentSumFunctorBP(nd4j::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentSumFunctorBP_, (context, input,
                indices, gradOut, output), FLOAT_TYPES, INDEXING_TYPES);
    }

    template <typename T, typename I>
    static int unsortedSegmentSumFunctorBP_(nd4j::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        if (input->isVector()) {
            Nd4jLong loop_size = input->lengthOf();
            auto numOfClasses = gradOut->lengthOf(); //indices->e<Nd4jLong>(loop_size - 1);
            segmentSumBPLinearKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(),
                    input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            auto packGradOut = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(gradOut->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            Nd4jLong* gradOutTads = packGradOut.specialShapeInfo();
            Nd4jLong* gradOutTadOffsets = packGradOut.specialOffsets();

            segmentSumBPTadKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                    gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
                    inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets,
                    outputTads, outputTadOffsets);
        }
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
        return Status::OK();
    }
    // -------------------------------------------------------------------------------------------------------------- //
    int unsortedSegmentSumFunctorBP(nd4j::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentSumFunctorBP_, (context, input, indices, gradOut, numOfClasses, output), FLOAT_TYPES, INDEXING_TYPES);
    }

}
}
}