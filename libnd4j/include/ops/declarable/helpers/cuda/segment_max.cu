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

            template<typename T, typename I>
            static __global__ void
            segmentMaxLinearKernel(void *input, Nd4jLong const* inputShape, int *starts, int *lengths, Nd4jLong numOfClasses,
                                   void *output, Nd4jLong const* outputShape) {
                __shared__                 T *val;
                __shared__                Nd4jLong xLen, zLen, zIndex;
                __shared__                T *x;
                __shared__                T *z;
                __shared__ int threadsPerSegment, start, finish;

                auto segment = blockIdx.x;
                if (threadIdx.x == 0) {
//                    threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
//                    segment = blockIdx.x / threadsPerSegment;
                    x = reinterpret_cast<T *>(input);
                    z = reinterpret_cast<T *>(output);
                    extern __shared__ unsigned char shmem[];
                    val = reinterpret_cast<T *>(shmem);
                    xLen = shape::length(inputShape);
                    zLen = shape::length(outputShape);

                    if (segment < numOfClasses) {
                        zIndex = shape::getIndexOffset(segment, outputShape);
                        start = starts[segment];
                        finish = start + lengths[segment];
                        z[zIndex] = x[shape::getIndexOffset(start, inputShape)];
                        val[segment] = z[zIndex];
                    }

                }
                __syncthreads();

                for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputShape);
                    sd::math::atomics::nd4j_atomicMax(&z[zIndex], x[xIndex]);
                }
            }
            // -------------------------------------------------------------------------------------------------------------- //

            template<typename T, typename I>
            static __global__ void
            unsortedSegmentMaxLinearKernel(void *input, Nd4jLong const* inputShape, void *indices, Nd4jLong const* indicesShape,
                                           int *starts, int *lengths, Nd4jLong numOfClasses, void *output,
                                           Nd4jLong const* outputShape) {
                __shared__                 T *val;
                __shared__                Nd4jLong xLen, zLen, zIndex;
                __shared__                T *x;
                __shared__                T *z;
                __shared__                I *y; //int threadsPerSegment, start, finish;
                auto segment = blockIdx.x;

                if (threadIdx.x == 0) {
                    x = reinterpret_cast<T *>(input);
                    z = reinterpret_cast<T *>(output);
                    y = reinterpret_cast<I *>(indices);
                    xLen = shape::length(inputShape);
                    zLen = shape::length(outputShape);

                    zIndex = shape::getIndexOffset(segment, outputShape);
                    //start = starts[segment];
                    //finish = start + lengths[segment];
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
                            sd::math::atomics::nd4j_atomicMax(&z[zIndex], x[xIndex]);
                        }
                    }
            }
            // -------------------------------------------------------------------------------------------------------------- //
            template <typename T, typename I>
            static __global__ void segmentMaxTadKernel(void* inputBuf, Nd4jLong const* inputShape, Nd4jLong const* inputTads,
                                                       Nd4jLong const* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf,
                                                       Nd4jLong const* outputShape, Nd4jLong const* outputTads, Nd4jLong const* outputTadOffsets, T filler = 0) {

                __shared__ T* val;
                __shared__ Nd4jLong len, zIndex, total;
                __shared__ T* z;
                __shared__ int start, finish;
                __shared__ I segment;

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
                if (idx <= total) {
                    auto x = reinterpret_cast<T *>(inputBuf) + inputTadOffsets[idx];
                    if (blockIdx.x == start) {
                        for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                            auto xIndex = shape::getIndexOffset(e, inputTads);
                            auto zIndex = shape::getIndexOffset(e, outputTads);
                            sd::math::atomics::nd4j_atomicMax(&z[zIndex], x[xIndex]);
                            //z[zIndex] = x[xIndex];
                        }
                    }
                    else {
                        for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                            auto xIndex = shape::getIndexOffset(e, inputTads);
                            auto zIndex = shape::getIndexOffset(e, outputTads);
                            if (lengths[segment])
                                sd::math::atomics::nd4j_atomicMax(&z[zIndex], x[xIndex]);
                        }
                    }
                }
            }
            // -------------------------------------------------------------------------------------------------------------- //

            template <typename T, typename I>
            static void segmentMaxFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
                //int numClasses = output->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                //Nd4jLong idx = indices->e<Nd4jLong>(0);
                output->assign(-DataTypeUtils::infOrMax<T>());
                auto stream = context->getCudaStream();
                indices->syncToHost();
                Nd4jLong numOfClasses = indices->e<Nd4jLong>(indices->lengthOf() - 1) + 1;
                NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses}, context);
                NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses}, context);

                classesRangesBegs.assign(indices->lengthOf());
                classesRangesLens.assign(0);
                dim3 dims(256, 512, 256);
                int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
                int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
                fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);

                NDArray::prepareSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});

                if (input->isVector()) {

                    segmentMaxLinearKernel<T,I><<<numOfClasses, input->lengthOf(), numOfClasses * 32 + 32, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
                }
                else {
                    std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
                    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->shapeInfo(), dimensions);
                    auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), dimensions);
                    auto inputTads = packX.specialShapeInfo();
                    auto inputTadOffsets = packX.specialOffsets();
                    auto outputTads = packZ.specialShapeInfo();
                    auto outputTadOffsets = packZ.specialOffsets();
                    segmentMaxTadKernel<T,I><<<packX.numberOfTads(), 512, 2048, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
                }
                NDArray::registerSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});
            }
            // -------------------------------------------------------------------------------------------------------------- //
            void segmentMaxFunctor(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* output) {
                NDArray::prepareSpecialUse({output}, {input, indices});
                BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentMaxFunctor_, (context, input, indices, output), NUMERIC_TYPES, INDEXING_TYPES);
                NDArray::registerSpecialUse({output}, {input, indices});
            }
            // -------------------------------------------------------------------------------------------------------------- //

            template <typename T, typename I>
            static void unsortedSegmentMaxFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
                output->assign(DataTypeUtils::infOrMax<T>());

                NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses}, context);
                NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses}, context);
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(sd::BroadcastOpsTuple::Assign(), row, classes);
                classesRangesBegs.assign(indices->lengthOf());
                classesRangesLens.assign(0);
                dim3 dims(numOfClasses, indices->lengthOf(), numOfClasses * 32 + 32);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
                fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
                int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
                int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());

                if (input->isVector()) {
                    unsortedSegmentMaxLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
                }
                else {
                    std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
                    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->shapeInfo(), dimensions);
                    auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), dimensions);
                    auto inputTads = packX.specialShapeInfo();
                    auto inputTadOffsets = packX.specialOffsets();
                    auto outputTads = packZ.specialShapeInfo();
                    auto outputTadOffsets = packZ.specialOffsets();
                    dims.x = input->sizeAt(0);
                    output->assign(-DataTypeUtils::max<T>());
                    segmentMaxTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
                }

            }
            // -------------------------------------------------------------------------------------------------------------- //
            void unsortedSegmentMaxFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                NDArray::prepareSpecialUse({output}, {input, indices});
                output->nullify();
                BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMaxFunctor_, (context, input, indices, numOfClasses, output), NUMERIC_TYPES, INDEXING_TYPES);
                NDArray::registerSpecialUse({output}, {input, indices});
            }

            // -------------------------------------------------------------------------------------------------------------- //
            // segment max
            // -------------------------------------------------------------------------------------------------------------- //
            template <typename T, typename I>
            static __global__ void segmentMaxBPLinearKernel(void* inputBuf, Nd4jLong const*  inputShape, void* forwardOutput,
                                                            Nd4jLong const*  forwardShape, void* eps, Nd4jLong const*  epsShape, void* indicesBuf, Nd4jLong const*  indicesShape,
                                                            void* outputBuf, Nd4jLong const*  outputShape) {
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

                    if (sd::math::nd4j_abs(gradIn[gradOffsetI] - x[xOffset]) <= T(1.e-6)) {
                        z[zOffset] = gradOut[gradOffsetO];
                    }
                }
            }

            // -------------------------------------------------------------------------------------------------------------- //
            template <typename T, typename I>
            static __global__ void segmentMaxBPTadKernel(void* inputBuf, Nd4jLong const*  inputShape, void* forwardOutput,
                                                         Nd4jLong const*  forwardShape, void* eps, Nd4jLong const*  epsShape, void* indicesBuf, Nd4jLong const*  indicesShape,
                                                         void* outputBuf, Nd4jLong const*  outputShape,Nd4jLong const*  inputTad,
                                                         Nd4jLong const*  inputOffsets, Nd4jLong const*  gradInTad, Nd4jLong const*  gradInOffsets,
                                                         Nd4jLong const*  gradOutTad, Nd4jLong const*  gradOutOffsets, Nd4jLong const*  outTad,
                                                         Nd4jLong const*  outOffsets) {
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
                        if (sd::math::nd4j_abs(in[e] - current[e]) <= T(1.e-6))
                            currentOut[e] = outGrad[e];
                    }
                }
            }
            // -------------------------------------------------------------------------------------------------------------- //
            template <typename T, typename I>
            int segmentMaxFunctorBP_(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
                //int numOfClasses = gradOut->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                auto stream = context->getCudaStream();
                NDArray tempRes(gradOut->ordering(), gradOut->getShapeAsVector(), DataTypeUtils::fromT<T>(), context);//->shapeInfo(), context);
                segmentMaxFunctor_<T, I>(context, input, indices, &tempRes);
                NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
                if (input->isVector()) {
                    Nd4jLong loop_size = input->lengthOf();
                    auto numOfClasses = gradOut->lengthOf(); //indices->e<Nd4jLong>(loop_size - 1);
                    segmentMaxBPLinearKernel<T,I><<<1 + gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                            tempRes.specialBuffer(), tempRes.specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                            indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
                }
                else {
                    std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
                    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->shapeInfo(), dimensions);
                    auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), dimensions);
                    auto packGradIn = sd::ConstantTadHelper::getInstance()->tadForDimensions(tempRes.shapeInfo(), dimensions);
                    auto packGradOut = sd::ConstantTadHelper::getInstance()->tadForDimensions(gradOut->shapeInfo(), dimensions);
                    Nd4jLong const*  inputTads = packX.specialShapeInfo();
                    Nd4jLong const*  inputTadOffsets = packX.specialOffsets();
                    Nd4jLong const*  outputTads = packZ.specialShapeInfo();
                    Nd4jLong const*  outputTadOffsets = packZ.specialOffsets();
                    Nd4jLong const*  gradInTads = packGradIn.specialShapeInfo();
                    Nd4jLong const*  gradInTadOffsets = packGradIn.specialOffsets();
                    Nd4jLong const*  gradOutTads = packGradOut.specialShapeInfo();
                    Nd4jLong const*  gradOutTadOffsets = packGradOut.specialOffsets();

                    segmentMaxBPTadKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                            tempRes.specialBuffer(), tempRes.specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                            indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
                            inputTads, inputTadOffsets, gradInTads, gradInTadOffsets, gradOutTads, gradOutTadOffsets,
                            outputTads, outputTadOffsets);
                }
                NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
                return Status::OK();
            }
            // -------------------------------------------------------------------------------------------------------------- //
            int segmentMaxFunctorBP(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
                NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
                BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentMaxFunctorBP_, (context, input,
                        indices, gradOut, output), FLOAT_TYPES, INDEXING_TYPES);
                NDArray::registerSpecialUse({output}, {input, indices, gradOut});
            }

            // -------------------------------------------------------------------------------------------------------------- //
            template <typename T, typename I>
            static int unsortedSegmentMaxFunctorBP_(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
                //int numOfClasses = gradOut->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                auto stream = context->getCudaStream();
                NDArray tempRes(gradOut->ordering(), gradOut->getShapeAsVector(), DataTypeUtils::fromT<T>(), context);//->shapeInfo(), context);
                unsortedSegmentMaxFunctor_<T, I>(context, input, indices, numOfClasses, &tempRes);
                NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
                if (input->isVector()) {
                    Nd4jLong loop_size = input->lengthOf();
                    auto numOfClasses = gradOut->lengthOf(); //indices->e<Nd4jLong>(loop_size - 1);
                    segmentMaxBPLinearKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                            tempRes.specialBuffer(), tempRes.specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                            indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
                }
                else {
                    std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
                    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->shapeInfo(), dimensions);
                    auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), dimensions);
                    auto packGradIn = sd::ConstantTadHelper::getInstance()->tadForDimensions(tempRes.shapeInfo(), dimensions);
                    auto packGradOut = sd::ConstantTadHelper::getInstance()->tadForDimensions(gradOut->shapeInfo(), dimensions);
                    Nd4jLong const*  inputTads = packX.specialShapeInfo();
                    Nd4jLong const*  inputTadOffsets = packX.specialOffsets();
                    Nd4jLong const*  outputTads = packZ.specialShapeInfo();
                    Nd4jLong const*  outputTadOffsets = packZ.specialOffsets();
                    Nd4jLong const*  gradInTads = packGradIn.specialShapeInfo();
                    Nd4jLong const*  gradInTadOffsets = packGradIn.specialOffsets();
                    Nd4jLong const*  gradOutTads = packGradOut.specialShapeInfo();
                    Nd4jLong const*  gradOutTadOffsets = packGradOut.specialOffsets();

                    segmentMaxBPTadKernel<T,I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                            tempRes.specialBuffer(), tempRes.specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
                            indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
                            inputTads, inputTadOffsets, gradInTads, gradInTadOffsets, gradOutTads, gradOutTadOffsets,
                            outputTads, outputTadOffsets);
                }
                NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
                return Status::OK();
            }
            // -------------------------------------------------------------------------------------------------------------- //
            int unsortedSegmentMaxFunctorBP(sd::LaunchContext* context , NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
                NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
                BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentMaxFunctorBP_, (context, input, indices, gradOut, numOfClasses, output), FLOAT_TYPES, INDEXING_TYPES);
                NDArray::registerSpecialUse({output}, {input, indices, gradOut});
            }
        }
    }
}