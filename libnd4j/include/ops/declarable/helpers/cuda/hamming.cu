/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/hamming.h>

namespace sd {
    namespace ops {
        namespace helpers {
            template <typename X, typename Z>
            static _CUDA_G void _hammingKernel(const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, void *reductionBuffer, Nd4jLong length) {
                auto x = reinterpret_cast<const X*>(vx);
                auto y = reinterpret_cast<const X*>(vy);
                auto z = reinterpret_cast<Z*>(vz);

                __shared__ Nd4jLong shared[CUDA_BLOCK_SIZE];

                // we want to nullify temporary memory before accumulating intermediate results
                shared[threadIdx.x] = 0;

                auto tid = threadIdx.x + blockIdx.x * blockDim.x;
                for (Nd4jLong e = tid; e < length; e += blockDim.x * gridDim.x) {
                    auto _x = static_cast<unsigned long long>(x[shape::getIndexOffset(e, xShapeInfo)]);
                    auto _y = static_cast<unsigned long long>(y[shape::getIndexOffset(e, yShapeInfo)]);

                    // we save intermediate result into shared memory
                    shared[threadIdx.x] += __popcll(_x ^ _y);
                }
                __syncthreads();

                // now we accumulate values
                auto numItems = sd::math::nd4j_min<Nd4jLong>(blockDim.x, length);
                auto floorPow2 = numItems;
                if (floorPow2 & (floorPow2 - 1)) {

                    while (floorPow2 & (floorPow2 - 1))
                        floorPow2 &= floorPow2 - 1;

                    if (threadIdx.x >= floorPow2)
                        shared[threadIdx.x - floorPow2] = shared[threadIdx.x - floorPow2] + shared[threadIdx.x];

                    __syncthreads();
                }
                __syncthreads();

                for (Nd4jLong activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
                    if (threadIdx.x < activeThreads && threadIdx.x + activeThreads < numItems)
                        shared[threadIdx.x] = shared[threadIdx.x] + shared[threadIdx.x + activeThreads];

                    __syncthreads();
                }
                __syncthreads();

                // FIXME: do we really want atomicAdd on global memory here
                // and store them to output
                if (threadIdx.x == 0 && shared[0] > 0)
                    sd::math::atomics::nd4j_atomicAdd<Z>(&z[0], static_cast<Z>(shared[threadIdx.x]));
            }

            template <typename X, typename Z>
            static void _hamming(LaunchContext *context, NDArray &x, NDArray &y, NDArray &z) {
                _hammingKernel<X, Z><<<256, CUDA_BLOCK_SIZE, 1024, *context->getCudaStream()>>>(x.specialBuffer(), x.specialShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), z.specialBuffer(), nullptr, x.lengthOf());
            }

            void hamming(LaunchContext *context, NDArray &x, NDArray &y, NDArray &output) {
                NDArray::prepareSpecialUse({&output}, {&x, &y});
                BUILD_DOUBLE_SELECTOR(x.dataType(), output.dataType(), _hamming, (context, x, y, output), INTEGER_TYPES, INDEXING_TYPES);
                NDArray::registerSpecialUse({&output}, {&x, &y});
            }
        }
    }
}