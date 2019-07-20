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

#include <ops/declarable/helpers/hashcode.h>


namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            static __global__ void splitBufferToChuncks(T* buffer, Nd4jLong* tempBuffer, Nd4jLong numBlocks, Nd4jLong blockSize, Nd4jLong length) {

                for (int b = blockIdx.x; b < numBlocks; b += gridDim.x) {
                    auto blockBuffer = buffer + b * numBlocks;

                    Nd4jLong r = 1;
                    for (int e = threadIdx.x; e < blockSize && e + (b * numBlocks) < length; e += blockDim.x) {
                        auto v = longBytes<T>(blockBuffer[e]);
                        r = 31 * r + v;
                    }

                    tempBuffer[b] = r;
                }
            }

            template <typename T>
            static __global__ void internalHash(Nd4jLong* tempBuffer, Nd4jLong* tempResult, Nd4jLong numBlocks, Nd4jLong blockSize, Nd4jLong lastLength) {

                for (int b = blockIdx.x; b < numBlocks; b += gridDim.x) {
                    auto blockBuffer = tempBuffer + b * numBlocks;

                    Nd4jLong r = 1;
                    for (int e = threadIdx.x; e < blockSize && e + (b * numBlocks) < lastLength; e += blockDim.x) {
                        auto v = longBytes<T>(blockBuffer[e]);
                        r = 31 * r + v;
                    }

                    tempResult[b] = r;
                }

            }


            static __global__ void lastStep(Nd4jLong* resultBuf, Nd4jLong* tempBufferA, Nd4jLong* tempResult, Nd4jLong length, Nd4jLong blockSize) {
                if (threadIdx.x == 0) {

                    if (length <= blockSize)
                        *resultBuf = *tempBufferA;
                    else
                        *resultBuf = *tempResult;
                }
            }

            template <typename T>
            void hashCode_(LaunchContext *context, NDArray &array, NDArray &result) {
                auto blockSize = 32;
                auto stream = context->getCudaStream();
                array.syncToDevice();

                NDArray::prepareSpecialUse({&result}, {&array});
                auto length = array.lengthOf();
                int numBlocks = length / blockSize + ((length % blockSize == 0) ? 0 : 1);
                auto tempA = NDArrayFactory::create<Nd4jLong>('c', {numBlocks}, context);
                auto tempB = NDArrayFactory::create<Nd4jLong>('c', { numBlocks / blockSize + 1}, context);

                auto buffer = reinterpret_cast<T*>(array.specialBuffer()); //bufferAsT<T>();
                auto tempBufferA = reinterpret_cast<Nd4jLong*>(tempA.specialBuffer()); //bufferAsT<Nd4jLong>();
                auto tempBufferB = reinterpret_cast<Nd4jLong*>(tempB.specialBuffer()); //bufferAsT<Nd4jLong>();

                // default buffer is the first one, because it might be the last one in case of small arrays (< blockSize)
                auto tempBuffer = tempBufferA;
                auto tempResult = tempBufferB;

                // we divide array into 32 element chunks, and store intermediate results once
                splitBufferToChuncks<T><<<numBlocks, length, 1024, *stream>>>(buffer, tempBuffer, numBlocks, blockSize, length);

                // we replace pointer with intermediate one, and repeat only one chunk left
                int iterationCount = 0;
                while (numBlocks > 1) {
                    int lastLength = numBlocks;
                    numBlocks = lastLength / blockSize + ((lastLength % blockSize == 0) ? 0 : 1);


                    internalHash<Nd4jLong><<<numBlocks, lastLength, 1024, *stream>>>(tempBuffer, tempResult, numBlocks, blockSize, lastLength);


                    iterationCount++;
                    // swapping buffers
                    if (iterationCount % 2 == 0) {
                        tempBuffer = tempBufferA;
                        tempResult = tempBufferB;
                    } else {
                        tempBuffer = tempBufferB;
                        tempResult = tempBufferA;
                    }
                }

                //lastStep<Nd4jLong><<<1,1,128, *stream>>>(result.specialBuffer(), tempBufferA, tempResult, length, blockSize);
                tempA.syncToHost();
                tempB.syncToHost();
                result.assign((length <= blockSize?tempA.e(0) : tempB.e(0)));

                NDArray::registerSpecialUse({&result}, {&array});
            }

            void hashCode(LaunchContext *context, NDArray &array, NDArray &result) {
                BUILD_SINGLE_SELECTOR(array.dataType(), hashCode_, (context, array, result), LIBND4J_TYPES);
            }

            BUILD_SINGLE_TEMPLATE(template void hashCode_, (LaunchContext* context, NDArray& array, NDArray& result), LIBND4J_TYPES);
        }
    }
}

