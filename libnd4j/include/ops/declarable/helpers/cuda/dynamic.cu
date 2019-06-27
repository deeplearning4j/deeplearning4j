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
// @author raver119@gmail.com
//
#include <ops/declarable/helpers/dynamic.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace nd4j {
    namespace ops {
        namespace helpers {


            template <typename X, typename Y>
            static _CUDA_G void dynamicPartitionScalarKernel(void *vx, Nd4jLong *xShapeInfo, void *vi, Nd4jLong *iShapeInfo, void **vz, Nd4jLong **zShapeInfos, const Nd4jLong numOutputs) {
                auto x = reinterpret_cast<X*>(vx);
                auto i = reinterpret_cast<Y*>(vi);
                auto xLength = shape::length(xShapeInfo);
                auto iLength = shape::length(iShapeInfo);

                extern __shared__ char shmem[];
                __shared__ Y *rawIndices;
                __shared__ Y *trueIndices;

                if (threadIdx.x == 0) {
                    rawIndices = reinterpret_cast<Y*>(shmem);
                    trueIndices = rawIndices + blockDim.x;
                }
                __syncthreads();


                for (Nd4jLong o = blockIdx.x; o < numOutputs; o += gridDim.x) {
                    auto z = reinterpret_cast<X*>(vz[o]);

                    auto zShapeInfo = zShapeInfos[o];
                    auto zLength = shape::length(zShapeInfo);

                    // iLimit should be
                    auto iLimit = iLength <= blockIdx.x ? blockIdx.x : (iLength + (blockIdx.x - (iLength % blockIdx.x)));
                    int cnt = 0;

                    for (Nd4jLong e = threadIdx.x; e < iLimit; e += blockDim.x) {
                        // load set of indices into shared memory
                        if (e < iLength)
                            rawIndices[threadIdx.x] = i[shape::getIndexOffset(e, iShapeInfo, iLength)];
                        __syncthreads();

                        // now we need to find out where our actual updates will be mapped
                        // TODO: this can be improved obviously, by using prefix-sum like approach
                        if (threadIdx.x == 0) {
                            for (int f = 0; f < blockDim.x; f++) {
                                if (rawIndices[f] == static_cast<Y>(o))
                                    trueIndices[f] = cnt++;
                                else
                                    trueIndices[f] = -1;
                            }
                        }
                        __syncthreads();


                        // doing actual update
                        if (e < iLength)
                            if (trueIndices[threadIdx.x] >= 0)
                                z[trueIndices[threadIdx.x]] = x[shape::getIndexOffset(e, xShapeInfo, xLength)];

                        __syncthreads();
                    }
                }
            }

            template <typename X, typename Y>
            static _CUDA_G void dynamicPartitionTadKernel(void *vx, Nd4jLong *xTadShapeInfo, Nd4jLong *xTadOffsets, Nd4jLong xLength, void *vindices, Nd4jLong *iShapeInfo, Nd4jLong iLength, void **vz, Nd4jLong **zTadShapeInfos, Nd4jLong **zTadOffsets, Nd4jLong numOutputs) {
                auto x = reinterpret_cast<X*>(vx);
                auto indices = reinterpret_cast<Y*>(vindices);

                for (int i = blockIdx.x; i < numOutputs; i += gridDim.x) {
                    auto z = reinterpret_cast<X*>(vz[i]);

                    int outCnt = 0;

                    for (Nd4jLong e = 0; e < iLength; e++) {
                        if (indices[shape::getIndexOffset(e, iShapeInfo, iLength)] == i) {
                            auto dx = x + xTadOffsets[e];
                            auto dz = z + zTadOffsets[i][outCnt++];

                            for (int f = threadIdx.x; f < xLength; f += blockDim.x) {
                                dz[shape::getIndexOffset(f, zTadShapeInfos[i], xLength)] = dx[shape::getIndexOffset(f, xTadShapeInfo, xLength)];
                            }
                        }
                    }
                }
            }

            template <typename X, typename Y>
            static void _dynamicPartitionFunctor(nd4j::LaunchContext * context, NDArray const* input, NDArray const* indices, std::vector<NDArray*>& outputList) {
                std::vector<std::pair<NDArray *, int>> outputs(outputList.size());
                int sourceDimsLen = input->rankOf() - indices->rankOf();

                unsigned int outSize = outputList.size();

                PointersManager pm(context, "dynamicPartition");

                if (sourceDimsLen) {
                    std::vector<int> sourceDims(sourceDimsLen);

                    for (int i = sourceDimsLen; i > 0; i--)
                        sourceDims[sourceDimsLen - i] = input->rankOf() - i;

                    auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), sourceDims);

                    std::vector<void *> outBuffers(outSize);
                    std::vector<Nd4jLong *> tadShapes(outSize);
                    std::vector<Nd4jLong *> tadOffsets(outSize);
                    std::vector<Nd4jLong> numTads(outSize);

                    for (unsigned int i = 0; i < outSize; i++) {
                        outputs[i].first = outputList[i];
                        std::vector<int> outDims(outputs[i].first->rankOf() - 1);

                        int r = outputs[i].first->rankOf();

                        for (int k = 1; k < r; k++)
                            outDims[k - 1] = k;

                        auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(outputList.at(i)->getShapeInfo(), outDims);

                        outBuffers[i] = outputList.at(i)->getSpecialBuffer();
                        tadShapes[i] = packZ.platformShapeInfo();
                        tadOffsets[i] = packZ.platformOffsets();
                    }

                    auto dOutBuffers = reinterpret_cast<void **>(pm.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void *)));
                    auto dOutTadShapes = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(tadShapes.data(), tadShapes.size() * sizeof(Nd4jLong *)));
                    auto dOutTadOffsets = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(tadOffsets.data(), tadOffsets.size() * sizeof(Nd4jLong *)));

                    dynamicPartitionTadKernel<X,Y><<<256, 512, 1024, *context->getCudaStream()>>>(input->getSpecialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), shape::length(packX.primaryShapeInfo()), indices->getSpecialBuffer(), indices->getSpecialShapeInfo(), indices->lengthOf(), dOutBuffers, dOutTadShapes, dOutTadOffsets, outSize);

                } else {
                    auto numThreads = 256;
                    auto shmemSize = numThreads * sizeof(Y) * 2 + 1024;


                    std::vector<void *> outBuffers;
                    std::vector<Nd4jLong *> outShapes;

                    for (auto v:outputList) {
                        outBuffers.emplace_back(v->getSpecialBuffer());
                        outShapes.emplace_back(v->getSpecialShapeInfo());
                    }

                    auto dOutBuffers = reinterpret_cast<void **>(pm.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void *)));
                    auto dOutShapes = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(outShapes.data(), outShapes.size() * sizeof(Nd4jLong *)));


                    dynamicPartitionScalarKernel<X,Y><<<256, numThreads, shmemSize, *context->getCudaStream()>>>(input->getSpecialBuffer(), input->getSpecialShapeInfo(), indices->getSpecialBuffer(), indices-> getSpecialShapeInfo(), dOutBuffers, dOutShapes, outSize);
                }

                pm.synchronize();
            }


            template <typename X, typename Y>
            static _CUDA_G void dynamicStitchScalarKernel(void **vx, Nd4jLong **xShapeInfos, void **vindices, Nd4jLong **iShapeInfos, int inputSize, void *vz, Nd4jLong *zShapeInfo, Nd4jLong zLength) {
                auto z = reinterpret_cast<X*>(vz);

                for (int e = blockIdx.x; e < inputSize; e += gridDim.x) {
                    auto x = reinterpret_cast<X*>(vx[e]);
                    auto indices = reinterpret_cast<Y*>(vindices[e]);

                    auto xShapeInfo = xShapeInfos[e];
                    auto iShapeInfo = iShapeInfos[e];

                    auto iLength = shape::length(iShapeInfo);

                    for (int i = threadIdx.x; i < iLength; i += blockDim.x) {
                        auto idx = indices[shape::getIndexOffset(i, iShapeInfo, iLength)];
                        if (idx >= 0 && idx < zLength)
                            z[shape::getIndexOffset(idx, zShapeInfo, zLength)] = x[shape::getIndexOffset(i, xShapeInfo, iLength)];
                    }
                }
            }

            template <typename X, typename Y>
            static _CUDA_G void dynamicStitchTadKernel(void **vx, Nd4jLong **xTadShapeInfos, Nd4jLong **xTadOffsets, void **vindices, Nd4jLong **iShapeInfos, int inputSize, void *vz, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {
                auto bz = reinterpret_cast<X*>(vz);

                for (int e = blockIdx.x; e < inputSize; e += gridDim.x) {
                    auto indices = reinterpret_cast<Y*>(vindices[e]);
                    auto iShapeInfo = iShapeInfos[e];

                    auto iLength = shape::length(iShapeInfo);
                    auto zLength = shape::length(zTadShapeInfo);

                    auto xShapeInfo = xTadShapeInfos[e];
                    auto xLength = shape::length(xShapeInfo);

                    for (int i = 0; i < iLength; i++) {
                        auto idx = indices[shape::getIndexOffset(i, iShapeInfo, iLength)];

                        auto z = bz + zTadOffsets[idx];
                        auto x = reinterpret_cast<X*>(vx[e]) + xTadOffsets[e][i];

                        for (int f = threadIdx.x; f < zLength; f += blockDim.x) {
                            z[shape::getIndexOffset(f, zTadShapeInfo, zLength)] = x[shape::getIndexOffset(f, xShapeInfo, xLength)];
                        }

                        __syncthreads();
                    }
                }
            }

            template <typename X, typename Y>
            static int _dynamicStitchFunctor(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray* output){

                int inputSize = inputs.size();

                PointersManager pm(context, "dynamicStitch");

                if (output->isVector()) {
                    std::vector<void *> inputBuffers(inputSize);
                    std::vector<Nd4jLong *> inputShapes(inputSize);
                    std::vector<void *> indicesBuffers(inputSize);
                    std::vector<Nd4jLong *> indicesShapes(inputSize);

                    for (int e = 0; e < inputSize; e++) {
                        inputBuffers[e] = inputs.at(e)->getSpecialBuffer();
                        indicesBuffers[e] = indices.at(e)->getSpecialBuffer();

                        inputShapes[e] = inputs.at(e)->getSpecialShapeInfo();
                        indicesShapes[e] = indices.at(e)->getSpecialShapeInfo();
                    }

                    auto dInputBuffers = reinterpret_cast<void **>(pm.replicatePointer(inputBuffers.data(), inputSize * sizeof(void *)));
                    auto dIndicesBuffers = reinterpret_cast<void **>(pm.replicatePointer(indicesBuffers.data(), inputSize * sizeof(void *)));
                    auto dInputShapes = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(inputShapes.data(), inputSize * sizeof(Nd4jLong *)));
                    auto dIndicesShapes = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(indicesShapes.data(), inputSize * sizeof(Nd4jLong *)));

                    dynamicStitchScalarKernel<X,Y><<<256, 256, 1024, *context->getCudaStream()>>>(dInputBuffers, dInputShapes, dIndicesBuffers, dIndicesShapes, inputSize, output->specialBuffer(), output->specialShapeInfo(), output->lengthOf());
                } else {
                    std::vector<int> restDims(output->rankOf() - 1);
                    for (int i = restDims.size(); i > 0;  i--)
                        restDims[restDims.size() - i] = output->rankOf() - i;

                    auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), restDims);

                    std::vector<void *> inputBuffers(inputSize);
                    std::vector<Nd4jLong *> inputTadShapes(inputSize);
                    std::vector<Nd4jLong *> inputTadOffsets(inputSize);

                    std::vector<void *> indicesBuffers(inputSize);
                    std::vector<Nd4jLong *> indicesShapes(inputSize);

                    for (int e = 0; e < inputSize; e++) {
                        std::vector<int> sourceDims(inputs[e]->rankOf() - indices[e]->rankOf());
                        for (int i = sourceDims.size(); i > 0;  i--)
                            sourceDims[sourceDims.size() - i] = inputs[e]->rankOf() - i;

                        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(inputs[e]->getShapeInfo(), sourceDims);

                        indicesBuffers[e] = indices[e]->getSpecialBuffer();
                        indicesShapes[e] = indices[e]->getSpecialShapeInfo();

                        inputBuffers[e] = inputs[e]->getSpecialBuffer();
                        inputTadShapes[e] = packX.platformShapeInfo();
                        inputTadOffsets[e] = packX.platformOffsets();
                    }

                    auto dInputBuffers = reinterpret_cast<void **>(pm.replicatePointer(inputBuffers.data(), inputSize * sizeof(void *)));
                    auto dInputTadShapes = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(inputTadShapes.data(), inputSize * sizeof(Nd4jLong *)));
                    auto dInputTadOffsets = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(inputTadOffsets.data(), inputSize * sizeof(Nd4jLong *)));

                    auto dIndicesBuffers = reinterpret_cast<void **>(pm.replicatePointer(indicesBuffers.data(), inputSize * sizeof(void *)));
                    auto dIndicesShapes = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(indicesShapes.data(), inputSize * sizeof(Nd4jLong *)));

                    dynamicStitchTadKernel<X,Y><<<256, 256, 1024, *context->getCudaStream()>>>(dInputBuffers, dInputTadShapes, dInputTadOffsets, dIndicesBuffers, dIndicesShapes, inputSize, output->specialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets());
                }

                pm.synchronize();

                return Status::OK();
            }

            template <typename T>
            static void _dynamicPartitionFunctorBP(NDArray const* input, NDArray const* indices, std::vector<NDArray*> const& inputGradientList, std::vector<NDArray*>& outputList) {

            }

            void dynamicPartitionFunctor(nd4j::LaunchContext * context, NDArray const* input, NDArray const* indices, std::vector<NDArray*>& outputList) {
                auto xType = input->dataType();
                auto yType = indices->dataType();

                NDArray::prepareSpecialUse({}, {indices, input});

                BUILD_DOUBLE_SELECTOR(xType, yType, _dynamicPartitionFunctor, (context, input, indices, outputList), LIBND4J_TYPES, INTEGER_TYPES);

                NDArray::registerSpecialUse({}, {indices, input});

                for (auto v:outputList)
                    v->tickWriteDevice();
            }

            template <typename T>
            static int _dynamicStitchFunctorBP(std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray const* gradInput, std::vector<NDArray*>& outputList){
                throw std::runtime_error("Not umplemented yet");
            }

            int dynamicStitchFunctor(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray* output){
                auto xType = inputs.at(0)->dataType();
                auto yType = indices.at(0)->dataType();

                for (auto v:indices) {
                    v->syncToDevice();
                    v->tickReadDevice();
                }

                for (auto v:inputs) {
                    v->syncToDevice();
                    v->tickReadDevice();
                }

                NDArray::prepareSpecialUse({output}, {});


                BUILD_DOUBLE_SELECTOR(xType, yType, _dynamicStitchFunctor, (context, inputs, indices, output), LIBND4J_TYPES, INTEGER_TYPES);

                NDArray::registerSpecialUse({output}, {});

                return Status::OK();
            }

            int dynamicStitchFunctorBP(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray const* gradInput, std::vector<NDArray*>& outputList) {
                auto xType = inputs.at(0)->dataType();

                BUILD_SINGLE_SELECTOR(xType, return _dynamicStitchFunctorBP, (inputs, indices, gradInput, outputList), LIBND4J_TYPES);
            }

            void dynamicPartitionFunctorBP(nd4j::LaunchContext * context, NDArray const* input, NDArray const* indices, std::vector<NDArray*> const& inputGradientList, std::vector<NDArray*>& outputList) {
                auto xType = input->dataType();

                BUILD_SINGLE_SELECTOR(xType, _dynamicPartitionFunctorBP, (input, indices, inputGradientList, outputList), LIBND4J_TYPES);
            }

            BUILD_SINGLE_TEMPLATE(template void _dynamicPartitionFunctorBP, (NDArray const* input, NDArray const* indices, std::vector<NDArray*> const& inputGradientList, std::vector<NDArray*>& outputList);, LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template int _dynamicStitchFunctorBP, (std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray const* gradInput, std::vector<NDArray*>& outputList);, LIBND4J_TYPES);

            BUILD_DOUBLE_TEMPLATE(template void _dynamicPartitionFunctor, (nd4j::LaunchContext * context, NDArray const* input, NDArray const* indices, std::vector<NDArray*>& outputList);, LIBND4J_TYPES, INTEGER_TYPES);
            BUILD_DOUBLE_TEMPLATE(template int _dynamicStitchFunctor, (nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray* output);, LIBND4J_TYPES, INTEGER_TYPES);


        }
    }
}

