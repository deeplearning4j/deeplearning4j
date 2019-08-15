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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <NDArrayFactory.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            //////////////////////////////////////////////////////////////////////////
            template <typename T, typename Z>
            static __global__ void global_mergeMaxIndex_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
                auto output = reinterpret_cast<Z*>(voutput);

                const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                for (Nd4jLong e = tid; e < length; e += step) {
                    T mVal = -DataTypeUtils::max<T>();
                    Z mIdx(0);

                    for (int i = 0; i < numArrays; i++) {
                        auto x = reinterpret_cast<T*>(inArrs[i]);
                        auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);
                        auto val = x[shape::getIndexOffset(e, xShape, length)];;
                        if (mVal < val)
                            mIdx = static_cast<Z>(e);
                    }
                    __syncthreads();

                    output[shape::getIndexOffset(e, outputShape, length)] = mIdx;
                }
            }

            template <typename T, typename Z>
            static void mergeMaxIndex_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
                std::vector<void *> inBuffers(inArrs.size());
                std::vector<void *> inShapes(inArrs.size());

                for (int e = 0; e < inArrs.size(); e++) {
                    inBuffers[e] = inArrs[e]->getSpecialBuffer();
                    inShapes[e] = inArrs[e]->getSpecialShapeInfo();
                }

                PointersManager manager(context, "mergeMaxIndex");

                auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
                auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
                auto length = output.lengthOf();

                global_mergeMaxIndex_<T,Z><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

                manager.synchronize();
            }

            void mergeMaxIndex(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
                BUILD_DOUBLE_SELECTOR(inArrs[0]->dataType(), output.dataType(), mergeMaxIndex_, (context, inArrs, output), LIBND4J_TYPES, INTEGER_TYPES);
            }

            BUILD_DOUBLE_TEMPLATE(template void mergeMaxIndex_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES, INTEGER_TYPES);

            //////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void global_mergeMax_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
                auto output = reinterpret_cast<T*>(voutput);

                const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                for (Nd4jLong e = tid; e < length; e += step) {
                    T mVal = -DataTypeUtils::max<T>();

                    for (int i = 0; i < numArrays; i++) {
                        auto x = reinterpret_cast<T*>(inArrs[i]);
                        auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);
                        auto val = x[shape::getIndexOffset(e, xShape, length)];;
                        if (mVal < val)
                            mVal = val;
                    }
                    __syncthreads();

                    output[shape::getIndexOffset(e, outputShape, length)] = mVal;
                }
            }

            template<typename T>
            static void mergeMax_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
                std::vector<void *> inBuffers(inArrs.size());
                std::vector<void *> inShapes(inArrs.size());

                for (int e = 0; e < inArrs.size(); e++) {
                    inBuffers[e] = inArrs[e]->getSpecialBuffer();
                    inShapes[e] = inArrs[e]->getSpecialShapeInfo();
                }

                PointersManager manager(context, "mergeMax");

                auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
                auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
                auto length = output.lengthOf();

                global_mergeMax_<T><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

                manager.synchronize();
            }
            BUILD_SINGLE_TEMPLATE(template void mergeMax_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

            void mergeMax(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
                BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (context, inArrs, output), LIBND4J_TYPES);
            }

            //////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void global_mergeAvg_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
                auto output = reinterpret_cast<T*>(voutput);

                const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                for (Nd4jLong e = tid; e < length; e += step) {
                    T sum(0.0f);

                    for (int i = 0; i < numArrays; i++) {
                        auto x = reinterpret_cast<T*>(inArrs[i]);
                        auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);

                        sum += x[shape::getIndexOffset(e, xShape, length)];
                    }

                    output[shape::getIndexOffset(e, outputShape, length)] = sum / numArrays;
                }
            }

            template<typename T>
            static void mergeAvg_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
                std::vector<void *> inBuffers(inArrs.size());
                std::vector<void *> inShapes(inArrs.size());

                for (int e = 0; e < inArrs.size(); e++) {
                    inBuffers[e] = inArrs[e]->getSpecialBuffer();
                    inShapes[e] = inArrs[e]->getSpecialShapeInfo();
                }

                PointersManager manager(context, "mergeAvg");

                auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
                auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
                auto length = output.lengthOf();

                global_mergeAvg_<T><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

                manager.synchronize();
            }
            BUILD_SINGLE_TEMPLATE(template void mergeAvg_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

            void mergeAvg(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
                BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (context, inArrs, output), LIBND4J_TYPES);
            }

            //////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void global_mergeAdd_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
                auto output = reinterpret_cast<T*>(voutput);

                const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                for (Nd4jLong e = tid; e < length; e += step) {
                    T sum(0.0f);

                    for (int i = 0; i < numArrays; i++) {
                        auto x = reinterpret_cast<T*>(inArrs[i]);
                        auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);

                        sum += x[shape::getIndexOffset(e, xShape, length)];
                    }

                    output[shape::getIndexOffset(e, outputShape, length)] = sum;
                }
            }

            template<typename T>
            static void mergeAdd_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
                std::vector<void *> inBuffers(inArrs.size());
                std::vector<void *> inShapes(inArrs.size());

                for (int e = 0; e < inArrs.size(); e++) {
                    inBuffers[e] = inArrs[e]->getSpecialBuffer();
                    inShapes[e] = inArrs[e]->getSpecialShapeInfo();
                }

                PointersManager manager(context, "mergeAdd");

                auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
                auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
                auto length = output.lengthOf();

                global_mergeAdd_<T><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

                manager.synchronize();
            }
            BUILD_SINGLE_TEMPLATE(template void mergeAdd_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

            void mergeAdd(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
                BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (context, inArrs, output), LIBND4J_TYPES);
            }
        }
    }
}