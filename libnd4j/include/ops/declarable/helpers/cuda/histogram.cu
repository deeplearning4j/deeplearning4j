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

#include <ops/declarable/helpers/histogram.h>
#include <NDArrayFactory.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename X, typename Z>
            void _CUDA_G histogramKernel(void *xBuffer, Nd4jLong *xShapeInfo, void *zBuffer, Nd4jLong *zShapeInfo, void *allocationPointer, void *reductionPointer, Nd4jLong numBins, double min_val, double max_val) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                auto dx = reinterpret_cast<X*>(xBuffer);
                auto result = reinterpret_cast<Z*>(zBuffer);

                __shared__ Z *bins;
                __shared__ int length;
                __shared__ Z *reductor;
                if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    bins = (Z *) shmem;
                    reductor = ((Z *) allocationPointer) + (numBins * blockIdx.x);

                    length = shape::length(xShapeInfo);
                }
                __syncthreads();

                Z binSize = (max_val - min_val) / (numBins);

                for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                    bins[e] = (Z) 0.0f;
                }
                __syncthreads();

                for (int e = tid; e < length; e+= blockDim.x * gridDim.x) {
                    int idx = (int) ((dx[e] - min_val) / binSize);
                    if (idx < 0) idx = 0;
                    else if (idx >= numBins) idx = numBins - 1;

                    nd4j::math::atomics::nd4j_atomicAdd(&bins[idx], (Z) 1.0f);
                }
                __syncthreads();

                // transfer shared memory to reduction memory


                if (gridDim.x > 1) {
                    unsigned int *tc = (unsigned int *)reductionPointer;
                    __shared__ bool amLast;

                    for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                        reductor[e] = bins[e];
                    }
                    __threadfence();
                    __syncthreads();

                    if (threadIdx.x == 0) {
                        unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
                        amLast = (ticket == gridDim.x - 1);
                    }
                    __syncthreads();

                    if (amLast) {
                        tc[16384] = 0;

                        // nullify shared memory for future accumulation
                        for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                            bins[e] = (Z) 0.0f;
                        }

                        // accumulate reduced bins
                        for (int r = 0; r < gridDim.x; r++) {
                            Z *ptrBuf = ((Z *)allocationPointer) + (r * numBins);

                            for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                                bins[e] += ptrBuf[e];
                            }
                        }
                        __syncthreads();

                        // write them out to Z
                        for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                            result[e] = bins[e];
                        }
                    }
                } else {
                    // if there's only 1 block - just write away data
                    for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                        result[e] = bins[e];
                    }
                }
            }

            template <typename X, typename Z>
            static void histogram_(nd4j::LaunchContext *context, void *xBuffer, Nd4jLong *xShapeInfo, void *zBuffer, Nd4jLong *zShapeInfo, Nd4jLong numBins, double min_val, double max_val) {
                int numThreads = 256;
                int numBlocks = nd4j::math::nd4j_max<int>(256, nd4j::math::nd4j_min<int>(1, shape::length(xShapeInfo) / numThreads));
                int workspaceSize = numBlocks * numBins;
                auto tmp = NDArrayFactory::create<Z>('c',{workspaceSize});

                histogramKernel<X, Z><<<numBlocks, numThreads, 32768, *context->getCudaStream()>>>(xBuffer, xShapeInfo, zBuffer, zShapeInfo, tmp.getSpecialBuffer(), context->getReductionPointer(), numBins, min_val, max_val);

                cudaStreamSynchronize(*context->getCudaStream());
            }

            void histogramHelper(nd4j::LaunchContext *context, NDArray &input, NDArray &output) {
                Nd4jLong numBins = output.lengthOf();
                double min_val = input.reduceNumber(reduce::SameOps::Min).e<double>(0);
                double max_val = input.reduceNumber(reduce::SameOps::Max).e<double>(0);

                BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), histogram_, (context, input.specialBuffer(), input.specialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), numBins, min_val, max_val), LIBND4J_TYPES, INTEGER_TYPES);

                NDArray::registerSpecialUse({&output}, {&input});
            }
        }
    }
}