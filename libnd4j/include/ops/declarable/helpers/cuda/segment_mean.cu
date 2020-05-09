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
    // Segment ops linear kernels
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void segmentMeanLinearKernel(void* input, Nd4jLong const*  inputShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong const*  outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x / threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
//            extern __shared__ unsigned char shmem[];
//            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            //[zIndex] =
            if (segment < numOfClasses) {
                zIndex = shape::getIndexOffset(segment, outputShape);
                start = starts[segment];
                finish = start + lengths[segment];
                //val[segment] = ;
                z[zIndex] = T(x[shape::getIndexOffset(start, inputShape)] / lengths[segment]);
//                val[segment] = z[zIndex];
            }

        }
        __syncthreads();

        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
            auto xIndex = shape::getIndexOffset(e, inputShape);
            if (lengths[segment])
                sd::math::atomics::nd4j_atomicAdd(&z[zIndex], T(x[xIndex] / lengths[segment]));
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void unsortedSegmentMeanLinearKernel(void* input, Nd4jLong const*  inputShape, void* indices, Nd4jLong const*  indicesShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong const*  outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ I* y; //int threadsPerSegment, start, finish;
        auto segment = blockIdx.x;// /
        if (threadIdx.x == 0) {
//            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
//            threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            y = reinterpret_cast<I*>(indices);
//            extern __shared__ unsigned char shmem[];
//            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

//            if (segment < numOfClasses) {
            zIndex = shape::getIndexOffset(segment, outputShape);
            //start = starts[segment];
            //finish = start + lengths[segment];
            if (lengths[segment] > 0)
                z[zIndex] = T(x[shape::getIndexOffset(starts[segment], inputShape)] / T(lengths[segment]));
            else
                z[zIndex] = 0; //DataTypeUtils::max<T>();
//                val[segment] = z[zIndex];
//            }

        }
        __syncthreads();
        if (lengths[segment] > 0)
            for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputShape);
                auto yIndex = shape::getIndexOffset(e, indicesShape);
                if (y[yIndex] == segment && e != starts[segment]) {
                    sd::math::atomics::nd4j_atomicAdd(&z[zIndex], T(x[xIndex]/T(lengths[segment])));
                }
            }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    // SegmentMean kernel
    template <typename T, typename I>
    static __global__ void segmentMeanTadKernel(void* inputBuf, Nd4jLong const*  inputShape, Nd4jLong const*  inputTads, Nd4jLong const*  inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf, Nd4jLong const*  outputShape, Nd4jLong const*  outputTads, Nd4jLong const*  outputTadOffsets) {
        __shared__ T* val;
        __shared__ Nd4jLong len, zIndex, total;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;
        auto segment = indices[blockIdx.x]; // / threadsPerSegment;

        if (threadIdx.x == 0) {
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
                    auto xIndex = shape::getIndexOffset(e, inputTads);
                    auto zIndex = shape::getIndexOffset(e, outputTads);
                    sd::math::atomics::nd4j_atomicAdd(&z[zIndex], T(x[xIndex]/lengths[segment]));
                }
            }
            else {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads);
                    auto zIndex = shape::getIndexOffset(e, outputTads);
                    if (lengths[segment])
                        sd::math::atomics::nd4j_atomicAdd(&z[zIndex], T(x[xIndex]/lengths[segment]));
                }
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    // segmen mean
    template <typename T, typename I>
    static void segmentMeanFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
        auto stream = context->getCudaStream();
        Nd4jLong numClasses = indices->e<Nd4jLong>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses}, context);
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses}, context);

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        NDArray::prepareSpecialUse({output}, {input, indices});
        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);

        if (input->isVector()) {
            segmentMeanLinearKernel<T,I><<<numClasses, input->lengthOf(), numClasses * 32 + 32, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->shapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), dimensions);
            auto inputTads = packX.specialShapeInfo();
            auto inputTadOffsets = packX.specialOffsets();
            auto outputTads = packZ.specialShapeInfo();
            auto outputTadOffsets = packZ.specialOffsets();
            segmentMeanTadKernel<T,I><<<input->sizeAt(0), 512, 2048, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }
        NDArray::registerSpecialUse({output}, {input, indices});

    }
    // -------------------------------------------------------------------------------------------------------------- //
    void segmentMeanFunctor(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices});
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segmentMeanFunctor_, (context, input, indices, output), NUMERIC_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices});
    }

    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static void unsortedSegmentMeanFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
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

        if (input->isVector()) {
            unsortedSegmentMeanLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            output->assign(0);
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->shapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), dimensions);
            Nd4jLong const*  inputTads = packX.specialShapeInfo();
            Nd4jLong const*  inputTadOffsets = packX.specialOffsets();
            Nd4jLong const*  outputTads = packZ.specialShapeInfo();
            Nd4jLong const*  outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            segmentMeanTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //
    void unsortedSegmentMeanFunctor(sd::LaunchContext* context , NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices});
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMeanFunctor_, (context, input, indices, numOfClasses, output),
                              NUMERIC_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices});
    }

    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void segmentMeanBPLinearKernel(void* inputBuf, Nd4jLong const*  inputShape, void* eps, Nd4jLong const*  epsShape, void* indicesBuf, Nd4jLong const*  indicesShape,
                                                     int* lengths, void* outputBuf, Nd4jLong const*  outputShape) {
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

            z[zOffset] = T(gradOut[gradOffsetO] / float(lengths[classIndex]));
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void segmentMeanBPTadKernel(void* inputBuf, Nd4jLong const*  inputShape, void* eps, Nd4jLong const*  epsShape,
                                                  void* indicesBuf, Nd4jLong const*  indicesShape, int* lengths, void* outputBuf, Nd4jLong const*  outputShape,Nd4jLong const*  inputTad,
                                                  Nd4jLong const*  inputOffsets, Nd4jLong const*  gradOutTad, Nd4jLong const*  gradOutOffsets, Nd4jLong const*  outTad, Nd4jLong const*  outOffsets) {
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
                    currentOut[zIndex] = T(outGrad[gradIndex] / float(lengths[segment]));
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    // backrop for mean
    template <typename T, typename I>
    int segmentMeanFunctorBP_(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        auto numClasses = indices->e<int>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses}, context);
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses}, context);

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());

        if (input->isVector()) {
            Nd4jLong loop_size = input->lengthOf();
            auto numOfClasses = gradOut->lengthOf(); //indices->e<Nd4jLong>(loop_size - 1);
            segmentMeanBPLinearKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(),
                    input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->shapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), dimensions);
//            auto packGradIn = sd::ConstantTadHelper::getInstance()->tadForDimensions(tempRes.shapeInfo(), dimensions);
            auto packGradOut = sd::ConstantTadHelper::getInstance()->tadForDimensions(gradOut->shapeInfo(), dimensions);
            Nd4jLong const*  inputTads = packX.specialShapeInfo();
            Nd4jLong const*  inputTadOffsets = packX.specialOffsets();
            Nd4jLong const*  outputTads = packZ.specialShapeInfo();
            Nd4jLong const*  outputTadOffsets = packZ.specialOffsets();
            Nd4jLong const*  gradOutTads = packGradOut.specialShapeInfo();
            Nd4jLong const*  gradOutTadOffsets = packGradOut.specialOffsets();

            segmentMeanBPTadKernel<T,I><<<indices->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                    gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), lengths,
                    output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets,
                    outputTads, outputTadOffsets);
        }
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
        return Status::OK();
    }
    // -------------------------------------------------------------------------------------------------------------- //
    // segmen mean bp main
    int segmentMeanFunctorBP(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentMeanFunctorBP_, (context, input,
                indices, gradOut, output), FLOAT_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static int unsortedSegmentMeanFunctorBP_(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        auto numClasses = indices->e<int>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses}, context);
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses}, context);

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());

        if (input->isVector()) {
            Nd4jLong loop_size = input->lengthOf();
            auto numOfClasses = gradOut->lengthOf(); //indices->e<Nd4jLong>(loop_size - 1);
            segmentMeanBPLinearKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(),
                    input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                    indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->shapeInfo(), dimensions);
            auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), dimensions);
//            auto packGradIn = sd::ConstantTadHelper::getInstance()->tadForDimensions(tempRes.shapeInfo(), dimensions);
            auto packGradOut = sd::ConstantTadHelper::getInstance()->tadForDimensions(gradOut->shapeInfo(), dimensions);
            Nd4jLong const*  inputTads = packX.specialShapeInfo();
            Nd4jLong const*  inputTadOffsets = packX.specialOffsets();
            Nd4jLong const*  outputTads = packZ.specialShapeInfo();
            Nd4jLong const*  outputTadOffsets = packZ.specialOffsets();
            Nd4jLong const*  gradOutTads = packGradOut.specialShapeInfo();
            Nd4jLong const*  gradOutTadOffsets = packGradOut.specialOffsets();

            segmentMeanBPTadKernel<T,I><<<indices->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                    gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), lengths,
                    output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets,
                    outputTads, outputTadOffsets);
        }
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
        return Status::OK();
    }
    // -------------------------------------------------------------------------------------------------------------- //
    int unsortedSegmentMeanFunctorBP(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentMeanFunctorBP_, (context, input, indices, gradOut, numOfClasses, output), FLOAT_TYPES, INDEXING_TYPES);
        NDArray::registerSpecialUse({output}, {input, indices, gradOut});
    }

}
}
}