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
#include <array/NDArrayFactory.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace sd {
    namespace ops {
        namespace helpers {
            //////////////////////////////////////////////////////////////////////////
            template <typename T, typename Z>
            static __global__ void mergeMaxIndexCudaLauncher(void** inArrs, void** inShapes, const int numArrays, void* voutput, const Nd4jLong* outputShape, Nd4jLong length) {
                auto output = reinterpret_cast<Z*>(voutput);

                const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                for (Nd4jLong e = tid; e < length; e += step) {
                    T mVal = -DataTypeUtils::max<T>();
                    Z mIdx(0);

                    for (int i = 0; i < numArrays; i++) {
                        auto x = reinterpret_cast<T*>(inArrs[i]);
                        auto xShape = reinterpret_cast<Nd4jLong*>(inShapes[i]);
                        auto val = x[shape::getIndexOffset(e, xShape)];;
                        if (mVal < val) {
                            mIdx = static_cast<Z>(i);
                            mVal = val;
                        }
                    }
                    
                    output[shape::getIndexOffset(e, outputShape)] = mIdx;
                }
            }

            template <typename T, typename Z>
            static void mergeMaxIndex_(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
                
                int nArrSize = static_cast<int>(inArrs.size());
                std::vector<const void*> inBuffers(nArrSize), inShapes(nArrSize);

                for (int e = 0; e < nArrSize; e++) {
                    inBuffers[e] = inArrs[e]->specialBuffer();
                    inShapes[e] = inArrs[e]->specialShapeInfo();
                }

                PointersManager manager(context, "mergeMaxIndex");

                auto pInBuffers = reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
                auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));
                auto length = output.lengthOf();

                const int threadsPerBlock = MAX_NUM_THREADS / 2;
                const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

                mergeMaxIndexCudaLauncher<T, Z><<<blocksPerGrid, threadsPerBlock, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, nArrSize, output.specialBuffer(), output.specialShapeInfo(), length);

                manager.synchronize();
            }

            void mergeMaxIndex(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
                
                NDArray::prepareSpecialUse({ &output }, inArrs);
                
                BUILD_DOUBLE_SELECTOR(inArrs[0]->dataType(), output.dataType(), mergeMaxIndex_, (context, inArrs, output), LIBND4J_TYPES, INDEXING_TYPES);

                NDArray::registerSpecialUse({ &output }, inArrs);
            }


            //////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void mergeMaxCudaLauncher(void** inArrs, void** inShapes, const int numArrays, void* voutput, const Nd4jLong* outputShape, Nd4jLong length) {
                auto output = reinterpret_cast<T*>(voutput);

                const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                for (Nd4jLong e = tid; e < length; e += step) {
                    T mVal = -DataTypeUtils::max<T>();

                    for (int i = 0; i < numArrays; i++) {
                        auto x = reinterpret_cast<const T*>(inArrs[i]);
                        auto xShape = reinterpret_cast<const Nd4jLong*>(inShapes[i]);
                        auto val = x[shape::getIndexOffset(e, xShape)];;
                        if (mVal < val)
                            mVal = val;
                    }

                    output[shape::getIndexOffset(e, outputShape)] = mVal;
                }
            }

            template<typename T>
            static void mergeMax_(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
                
                int nArrsSize = static_cast<int>(inArrs.size());

                std::vector<const void*> inBuffers(nArrsSize), inShapes(nArrsSize);

                for (int e = 0; e < nArrsSize; e++) {
                    inBuffers[e] = inArrs[e]->specialBuffer();
                    inShapes[e] = inArrs[e]->specialShapeInfo();
                }

                PointersManager manager(context, "mergeMax");

                auto pInBuffers = reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
                auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));
                auto length = output.lengthOf();

                const int threadsPerBlock = MAX_NUM_THREADS / 2;
                const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

                mergeMaxCudaLauncher<T><<<blocksPerGrid, threadsPerBlock, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, nArrsSize, output.specialBuffer(), output.specialShapeInfo(), length);

                manager.synchronize();
            }

            void mergeMax(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
                
                NDArray::prepareSpecialUse({ &output }, inArrs);
                
                BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (context, inArrs, output), LIBND4J_TYPES);
                
                NDArray::registerSpecialUse({ &output }, inArrs);
            }

            //////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void mergeMaxBpCudaLauncher(
                    void** inArrs, void** inShapes,
                    const void* vgradient, const Nd4jLong* gradientShape,
                    const int numArrays,
                    void** outArrs, void** outShapes,
                    Nd4jLong length,
                    bool bSameOrderAndEws1) {

                auto grad = reinterpret_cast<const T*>(vgradient);

                const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                int coords[MAX_RANK];

                for (Nd4jLong e = tid; e < length; e += step) {

                    T mVal = -DataTypeUtils::max<T>();
                    int nMaxIndex = 0;
                    auto xOffset = e, zOffset = e, gradOffset = e;

                    if (!bSameOrderAndEws1) {
                        shape::index2coords(e, gradientShape, coords);
                        gradOffset = shape::getOffset(gradientShape, coords);
                    }

                    for (int i = 0; i < numArrays; i++) {
                        auto x = reinterpret_cast<T*>(inArrs[i]);

                        if (!bSameOrderAndEws1) {
                            auto xShape = reinterpret_cast<Nd4jLong*>(inShapes[i]);
                            xOffset = shape::getOffset(xShape, coords);
                        }

                        auto val = x[xOffset];
                        if (mVal < val) {
                            mVal = val;
                            nMaxIndex = i;
                        }
                    }
                  
                    // outputs have to be pre-nullify                 
                    if (!bSameOrderAndEws1) {
                        auto outShape = reinterpret_cast<Nd4jLong*>(outShapes[nMaxIndex]);
                        zOffset = shape::getOffset(outShape, coords);
                    }

                    auto output = reinterpret_cast<T*>(outArrs[nMaxIndex]);

                    output[zOffset] = grad[gradOffset];
                }
            }

            template<typename T>
            static void mergeMaxBp_(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, std::vector<NDArray*>& outArrs, int nArrSize, bool bSameOrderAndEws1) {

                std::vector<const void*> inBuffers(nArrSize), inShapes(nArrSize), outBuffers(nArrSize), outShapes(nArrSize);

                for (int e = 0; e < nArrSize; e++) {
                    inBuffers[e] = inArrs[e]->specialBuffer();
                    inShapes[e] = inArrs[e]->specialShapeInfo();
                    outBuffers[e] = outArrs[e]->specialBuffer();
                    outShapes[e] = outArrs[e]->specialShapeInfo();
                }

                PointersManager manager(context, "mergeMaxBp");

                auto pInBuffers = reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
                auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));

                auto pOutBuffers = reinterpret_cast<void**>(manager.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void*)));
                auto pOutShapes = reinterpret_cast<void**>(manager.replicatePointer(outShapes.data(), outShapes.size() * sizeof(void*)));

                auto length = inArrs[nArrSize]->lengthOf();

                const int threadsPerBlock = MAX_NUM_THREADS / 2;
                const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

                mergeMaxBpCudaLauncher<T><<<blocksPerGrid, threadsPerBlock, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, inArrs[nArrSize]->specialBuffer(),
                    inArrs[nArrSize]->specialShapeInfo(), nArrSize, pOutBuffers, pOutShapes,
                    length, bSameOrderAndEws1);
                
                manager.synchronize();
            }

            void mergeMaxBp(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, std::vector<NDArray*>& outArrs) {

                // not use gradient
                int nArrSize = static_cast<int>(inArrs.size() - 1);
                
                const std::vector<const NDArray*>& out = reinterpret_cast<const std::vector<const NDArray*>&>(outArrs);

                NDArray::prepareSpecialUse(out, inArrs);

                bool bSameOrderAndEws1 = (1 == inArrs[nArrSize]->ews());
                auto ordering = inArrs[nArrSize]->ordering();
                  
                for (int i = 0; i < nArrSize; ++i) {
                    bSameOrderAndEws1 &= (ordering == inArrs[i]->ordering());
                    bSameOrderAndEws1 &= (1 == inArrs[i]->ews());
                   
                    bSameOrderAndEws1 &= (ordering == outArrs[i]->ordering());
                    bSameOrderAndEws1 &= (1 == outArrs[i]->ews());
                }

                BUILD_SINGLE_SELECTOR(inArrs[nArrSize]->dataType(), mergeMaxBp_, (context, inArrs, outArrs, nArrSize, bSameOrderAndEws1), LIBND4J_TYPES);

                NDArray::registerSpecialUse( out, inArrs );
            }


            //////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void mergeAvgCudaLauncher(void** inArrs, void** inShapes, const int numArrays, void* voutput, const Nd4jLong* outputShape, Nd4jLong length) {
                auto output = reinterpret_cast<T*>(voutput);

                const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                for (Nd4jLong e = tid; e < length; e += step) {
                    T sum(0.0f);

                    for (int i = 0; i < numArrays; i++) {
                        auto x = reinterpret_cast<T*>(inArrs[i]);
                        auto xShape = reinterpret_cast<Nd4jLong*>(inShapes[i]);

                        sum += x[shape::getIndexOffset(e, xShape)];
                    }

                    output[shape::getIndexOffset(e, outputShape)] = sum / numArrays;
                }
            }

            template<typename T>
            static void mergeAvg_(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
                
                std::vector<const void*> inBuffers(inArrs.size()), inShapes(inArrs.size());

                for (int e = 0; e < inArrs.size(); e++) {
                    inBuffers[e] = inArrs[e]->specialBuffer();
                    inShapes[e] = inArrs[e]->specialShapeInfo();
                }

                PointersManager manager(context, "mergeAvg");

                auto pInBuffers = reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
                auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));
                auto length = output.lengthOf();

                const int threadsPerBlock = MAX_NUM_THREADS / 2;
                const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

                mergeAvgCudaLauncher<T><<<blocksPerGrid, threadsPerBlock, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int)inArrs.size(), output.specialBuffer(), output.specialShapeInfo(), length);

                manager.synchronize();
            }

            void mergeAvg(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
                
                NDArray::prepareSpecialUse({ &output }, inArrs);

                BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (context, inArrs, output), FLOAT_TYPES);

                NDArray::registerSpecialUse({ &output }, inArrs);
            }
            //////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void mergeAvgBpCudaLauncher(
                    const void* vgradient, const Nd4jLong* gradientShape,
                    void** outArrs, void** outShapes,
                    const int numArrays,
                    Nd4jLong length,
                    bool bSameOrderAndEws1) {

                auto grad = reinterpret_cast<const T*>(vgradient);

                const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                int coords[MAX_RANK];

                for (Nd4jLong e = tid; e < length; e += step) {

                    auto zOffset = e, gradOffset = e;
                    if (!bSameOrderAndEws1) {
                        shape::index2coords(e, gradientShape, coords);
                        gradOffset = shape::getOffset(gradientShape, coords);
                    }

                    for (int i = 0; i < numArrays; i++) {

                        if (!bSameOrderAndEws1) {
                            auto outShape = reinterpret_cast<Nd4jLong*>(outShapes[i]);
                            zOffset = shape::getOffset(outShape, coords);
                        }

                        auto output = reinterpret_cast<T*>(outArrs[i]);

                        output[zOffset] = grad[gradOffset] / numArrays;
                    }
                }
            }

            template<typename T>
            static void mergeAvgBp_(sd::LaunchContext* context, const NDArray& gradient, std::vector<NDArray*>& outArrs, bool bSameOrderAndEws1) {

                int nArrSize = static_cast<int>(outArrs.size());

                std::vector<const void*> outBuffers(nArrSize), outShapes(nArrSize);

                for (int e = 0; e < nArrSize; e++) {
                    outBuffers[e] = outArrs[e]->specialBuffer();
                    outShapes[e] = outArrs[e]->specialShapeInfo();
                }

                PointersManager manager(context, "mergeAvgBp");

                auto pOutBuffers = reinterpret_cast<void**>(manager.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void*)));
                auto pOutShapes = reinterpret_cast<void**>(manager.replicatePointer(outShapes.data(), outShapes.size() * sizeof(void*)));

                auto length = gradient.lengthOf();
                
                const int threadsPerBlock = MAX_NUM_THREADS / 2;
                const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

                mergeAvgBpCudaLauncher<T><<<blocksPerGrid, threadsPerBlock, 512, *context->getCudaStream()>>>(gradient.specialBuffer(), gradient.specialShapeInfo(),
                    pOutBuffers, pOutShapes, nArrSize, length, bSameOrderAndEws1);

                manager.synchronize();
            }

            void mergeAvgBp(sd::LaunchContext* context, const NDArray& gradient, std::vector<NDArray*>& outArrs) {

                const std::vector<const NDArray*>& out = reinterpret_cast<const std::vector<const NDArray*>&>(outArrs);

                NDArray::prepareSpecialUse( out, { &gradient });

                bool bSameOrderAndEws1 = (1 == gradient.ews());
                auto ordering = gradient.ordering();

                for (const auto& v : outArrs) {
                    bSameOrderAndEws1 &= (ordering == v->ordering());
                    bSameOrderAndEws1 &= (1 == v->ews());
                }

                BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAvgBp_, (context, gradient, outArrs, bSameOrderAndEws1), LIBND4J_TYPES);

                NDArray::prepareSpecialUse(out, { &gradient });
            }

            //////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void mergeAddCudaLauncher(void** inArrs, void** inShapes, const int numArrays, void* voutput, const Nd4jLong* outputShape, Nd4jLong length) {
                
                auto output = reinterpret_cast<T*>(voutput);

                const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                for (Nd4jLong e = tid; e < length; e += step) {
                    T sum(0.0f);

                    for (int i = 0; i < numArrays; i++) {
                        auto x = reinterpret_cast<T*>(inArrs[i]);
                        auto xShape = reinterpret_cast<Nd4jLong*>(inShapes[i]);

                        sum += x[shape::getIndexOffset(e, xShape)];
                    }

                    output[shape::getIndexOffset(e, outputShape)] = sum;
                }
            }

            template<typename T>
            static void mergeAdd_(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
                
                int nArrSize = static_cast<int>(inArrs.size());
                std::vector<const void*> inBuffers(nArrSize), inShapes(nArrSize);

                for (int e = 0; e < nArrSize; e++) {
                    inBuffers[e] = inArrs[e]->specialBuffer();
                    inShapes[e] = inArrs[e]->specialShapeInfo();
                }

                PointersManager manager(context, "mergeAdd");

                auto pInBuffers = reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
                auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));
                auto length = output.lengthOf();

                const int threadsPerBlock = MAX_NUM_THREADS / 2;
                const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

                mergeAddCudaLauncher<T><<<blocksPerGrid, threadsPerBlock, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, nArrSize, output.specialBuffer(), output.specialShapeInfo(), length);

                manager.synchronize();
            }
            BUILD_SINGLE_TEMPLATE(template void mergeAdd_, (sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output), NUMERIC_TYPES);

            void mergeAdd(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
                
                NDArray::prepareSpecialUse({ &output }, inArrs);
                
                BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (context, inArrs, output), NUMERIC_TYPES);

                NDArray::registerSpecialUse({ &output }, inArrs);
            }

            //////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void mergeAddBpCudaLauncher(const void* vgradient, const Nd4jLong* gradientShape, void** outArrs, void** outShapes,
                const int numArrays, Nd4jLong length, bool bSameOrderAndEws1) {

                auto grad = reinterpret_cast<const T*>(vgradient);

                const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
                const auto step = gridDim.x * blockDim.x;

                int coords[MAX_RANK];

                for (Nd4jLong e = tid; e < length; e += step) {

                    auto zOffset = e, gradOffset = e;
                    if (!bSameOrderAndEws1) {
                        shape::index2coords(e, gradientShape, coords);
                        gradOffset = shape::getOffset(gradientShape, coords);
                    }

                    for (int i = 0; i < numArrays; i++) {
                        
                        if (!bSameOrderAndEws1) {
                            auto outShape = reinterpret_cast<Nd4jLong*>(outShapes[i]);
                            zOffset = shape::getOffset(outShape, coords);
                        }

                        auto output = reinterpret_cast<T*>(outArrs[i]);

                        output[zOffset] = grad[gradOffset];
                    }
                }
            }

            template<typename T>
            static void mergeAddBp_(sd::LaunchContext* context, const NDArray& gradient, std::vector<NDArray*>& outArrs, bool bSameOrderAndEws1) {

                int nArrSize = static_cast<int>(outArrs.size());

                std::vector<const void*> outBuffers(nArrSize), outShapes(nArrSize);

                for (int e = 0; e < nArrSize; e++) {
                    outBuffers[e] = outArrs[e]->specialBuffer();
                    outShapes[e] = outArrs[e]->specialShapeInfo();
                }

                PointersManager manager(context, "mergeAddBp");

                auto pOutBuffers = reinterpret_cast<void**>(manager.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void*)));
                auto pOutShapes = reinterpret_cast<void**>(manager.replicatePointer(outShapes.data(), outShapes.size() * sizeof(void*)));

                auto length = gradient.lengthOf();

                const int threadsPerBlock = MAX_NUM_THREADS / 2;
                const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

                mergeAddBpCudaLauncher<T><<<blocksPerGrid, threadsPerBlock, 512, *context->getCudaStream()>>>(gradient.specialBuffer(), gradient.specialShapeInfo(),
                    pOutBuffers, pOutShapes, nArrSize, length, bSameOrderAndEws1);

                manager.synchronize();
            }

            void mergeAddBp(sd::LaunchContext* context, const NDArray& gradient, std::vector<NDArray*>& outArrs) {

                const std::vector<const NDArray*>& out = reinterpret_cast<const std::vector<const NDArray*>& >(outArrs);
                NDArray::prepareSpecialUse( out, { &gradient });

                bool bSameOrderAndEws1 = (1 == gradient.ews());
                auto ordering = gradient.ordering();

                for (const auto& v : outArrs) {
                    bSameOrderAndEws1 &= (ordering == v->ordering());
                    bSameOrderAndEws1 &= (1 == v->ews());
                }

                BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAddBp_, (context, gradient, outArrs, bSameOrderAndEws1), LIBND4J_TYPES);

                NDArray::prepareSpecialUse( out, { &gradient });
            }

        }
    }
}
