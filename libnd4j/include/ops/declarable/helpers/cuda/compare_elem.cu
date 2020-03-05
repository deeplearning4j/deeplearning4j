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

#include <ops/declarable/helpers/compare_elem.h>

namespace sd {
    namespace ops {
        namespace helpers {

            template <typename T>
            static _CUDA_G void comparator(void *vx, const Nd4jLong *xShapeInfo, Nd4jLong length, const bool isStrict, void *reductionBuffer, bool *z) {
                auto x = reinterpret_cast<T*>(vx);
                auto reduction = reinterpret_cast<uint32_t*>(reductionBuffer);

                extern __shared__ uint32_t shared[];
                auto tid = threadIdx.x + blockIdx.x * blockDim.x;

                shared[threadIdx.x] = 0;

                // each thread will compare 2 elements: E and E+1
                for (int e = tid; e < length - 1; e += blockDim.x * gridDim.x) {
                    auto val0 = x[shape::getIndexOffset(e, xShapeInfo)];
                    auto val1 = x[shape::getIndexOffset(e+1, xShapeInfo)];

                    bool v = false;
                    if (isStrict)
                        v = val1 > val0;
                    else
                        v = val1 >= val0;

                    // store comparison result in shared memory
                    shared[threadIdx.x] += v ? 0 : 1;
                }
                __syncthreads();

                // aggregate sums in shared memory
                for (uint activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
                    if (threadIdx.x < activeThreads)
                        shared[threadIdx.x] += shared[threadIdx.x + activeThreads];
                    __syncthreads();
                }


                // store over the grid if we have more than 1 block
                if (gridDim.x > 1) {

                    auto tc = reinterpret_cast<unsigned int *>(reductionBuffer);
                    __shared__ bool amLast;

                    tid = threadIdx.x;
                    if (threadIdx.x == 0)
                        reduction[blockIdx.x] = shared[0];

                    __threadfence();
                    __syncthreads();

                    if (threadIdx.x == 0) {
                        unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
                        amLast = (ticket == gridDim.x - 1);
                    }

                    __syncthreads();

                    if (amLast) {
                        tc[16384] = 0;
                        shared[threadIdx.x] = 0;

                        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
                            shared[threadIdx.x] += reduction[i];

                        __syncthreads();

                        for (uint activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
                            if (threadIdx.x < activeThreads)
                                shared[threadIdx.x] += shared[threadIdx.x + activeThreads];
                            __syncthreads();
                        }

                        __syncthreads();

                        if (threadIdx.x == 0) {
                            z[0] = shared[0] == 0;
                        }
                    }
                }
                else {
                    // if we have only 1 block, we just store results right away
                    if (threadIdx.x == 0) {
                        auto tc = reinterpret_cast<unsigned int*>(reductionBuffer);
                        tc[16384] = 0;
                        z[0] = shared[0] == 0;
                    }
                }
            }

            template<typename T>
            static void _compare_elem(sd::LaunchContext * context, NDArray *input, bool isStrictlyIncreasing, bool& output) {
                auto z = NDArrayFactory::create<bool>(false, context);

                const int numThreads = 256;
                const int numBlocks = sd::math::nd4j_min<int>(128, sd::math::nd4j_max<int>(1, input->lengthOf() / numThreads));

                comparator<T><<<numBlocks, numThreads, numThreads * 4 + 1024, *context->getCudaStream()>>>(input->specialBuffer(), input->specialShapeInfo(), input->lengthOf(), isStrictlyIncreasing, context->getReductionPointer(), reinterpret_cast<bool *>(z.specialBuffer()));

                z.tickWriteDevice();
                sd::DebugHelper::checkErrorCode(context->getCudaStream(), "is_strictly_increasing");

                output = z.e<bool>(0);
            }

            void compare_elem(sd::LaunchContext * context, NDArray *input, bool isStrictlyIncreasing, bool& output) {
                auto xType = input->dataType();
                input->syncToDevice();

                BUILD_SINGLE_SELECTOR(xType, _compare_elem, (context, input, isStrictlyIncreasing, output), LIBND4J_TYPES);
            }


            BUILD_SINGLE_TEMPLATE(template void _compare_elem, (sd::LaunchContext * context, NDArray *A, bool isStrictlyIncreasing, bool& output);, LIBND4J_TYPES);
        }
    }
}
