/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

#include <ops/declarable/helpers/threshold.h>
#include <loops/type_conversions.h>
#include <helpers/PointersManager.h>
#include <vector>

namespace sd {
    namespace ops {
        namespace helpers {
            void prescanArrayRecursive(int** g_scanBlockSums, int *dZ, int *dX, int numElements, int level) {
                auto stream = LaunchContext::defaultContext()->getCudaStream();


                int blockSize = 512; // max size of the thread blocks
                int numBlocks = sd::math::nd4j_max<int>(1, static_cast<int>(ceil(static_cast<float>(numElements) / (2.f * blockSize))));
                int numThreads;

                if (numBlocks > 1)
                    numThreads = blockSize;
                else if (sd::isPowerOfTwo(numElements))
                    numThreads = numElements / 2;
                else
                    numThreads = sd::floorPow2(numElements);

                int numEltsPerBlock = numThreads * 2;

                // if this is a non-power-of-2 array, the last block will be non-full
                // compute the smallest power of 2 able to compute its scan.
                int numEltsLastBlock =
                        numElements - (numBlocks-1) * numEltsPerBlock;
                int numThreadsLastBlock = sd::math::nd4j_max<int>(1, numEltsLastBlock / 2);
                int np2LastBlock = 0;
                int sharedMemLastBlock = 0;

                if (numEltsLastBlock != numEltsPerBlock) {
                    np2LastBlock = 1;

                    if(!isPowerOfTwo(numEltsLastBlock))
                        numThreadsLastBlock = floorPow2(numEltsLastBlock);

                    unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
                    sharedMemLastBlock = sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
                }

                // padding space is used to avoid shared memory bank conflicts
                int extraSpace = numEltsPerBlock / NUM_BANKS;
                int sharedMemSize = sizeof(int) * (numEltsPerBlock + extraSpace);

                // setup execution parameters
                // if NP2, we process the last block separately
                dim3 grid(sd::math::nd4j_max<int>(1, numBlocks - np2LastBlock), 1, 1);
                dim3 threads(numThreads, 1, 1);
                dim3 gridOnes(1, 1, 1);
                dim3 threadsOnes(numThreadsLastBlock, 1, 1);

                if (sharedMemSize < 2048)
                    sharedMemSize = 2048;

                if (sharedMemLastBlock < 2048)
                    sharedMemLastBlock = 2048;

                // execute the scan
                if (numBlocks > 1) {
                    sd::prescanLauncher<true, false>(grid, threads, sharedMemSize, stream, dZ, dX, g_scanBlockSums[level], numThreads * 2, 0, 0);
                    if (np2LastBlock) {
                        sd::prescanLauncher<true, true>(gridOnes, threadsOnes, sharedMemLastBlock, stream, dZ, dX, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
                    }

                    // After scanning all the sub-blocks, we are mostly done.  But now we
                    // need to take all of the last values of the sub-blocks and scan those.
                    // This will give us a new value that must be sdded to each block to
                    // get the final results.
                    // recursive (CPU) call
                    prescanArrayRecursive(g_scanBlockSums, g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level+1);

                    sd::uniformAdd<<<grid, threads, 1024, *stream>>>(dZ, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);

                    if (np2LastBlock) {
                        sd::uniformAdd<<<1, numThreadsLastBlock, 1024, *stream>>>(dZ, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
                    }
                } else if (isPowerOfTwo(numElements)) {
                    sd::prescanLauncher<false, false>(grid, threads, sharedMemSize, stream, dZ, dX, 0, numThreads * 2, 0, 0);
                } else {
                    sd::prescanLauncher<false, true>(grid, threads, sharedMemSize, stream, dZ, dX, 0, numElements, 0, 0);
                }

                sd::DebugHelper::checkErrorCode(stream, "prescanArray(...) failed");
            }

            static void encodeThresholdP2Int_(void **prs, int *dx, Nd4jLong N, int *dz) {
                auto stream = LaunchContext::defaultContext()->getCudaStream();

                prescanArrayRecursive(reinterpret_cast<int**>(prs), dz, dx + 1, (int) N, 0);
                sd::DebugHelper::checkErrorCode(stream, "encodeThresholdP2Int(...) failed");
            }

            static void encodeThresholdP3_(void *dx, const Nd4jLong *hXShapeInfo, int *offsets, Nd4jLong N, int *dz){
                auto stream = LaunchContext::defaultContext()->getCudaStream();

                int blockSize = 512;
                int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

                dim3 launchDims(numBlocks, blockSize, 8192);
                auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
                BUILD_SINGLE_SELECTOR(xType, encoderKernelP3Generic, (launchDims, stream, dx, offsets, N, dz), FLOAT_TYPES);

                sd::DebugHelper::checkErrorCode(stream, "encodeThresholdP3Float(...) failed");
            }


            static NDArray thresholdEstimate_(const NDArray &updates, const float threshold) {
                const int numThreads = 512;
                const int numBlocks = updates.lengthOf() / numThreads + (updates.lengthOf() % numThreads ? 1 : 0);

                auto tmp = NDArrayFactory::create<int>('c', {numBlocks + 1});

                dim3 launchDims(numBlocks, numThreads, 1024);
                auto xType = updates.dataType();

                NDArray::prepareSpecialUse({&tmp}, {&updates});
                BUILD_SINGLE_SELECTOR(xType, encoderKernelP1Generic, (launchDims, LaunchContext::defaultContext()->getCudaStream(), updates.specialBuffer(), updates.lengthOf(), tmp.specialBuffer(), threshold), FLOAT_TYPES);
                NDArray::registerSpecialUse({&tmp}, {&updates});

                return std::move(tmp);
            }

            int32_t thresholdEstimate(const NDArray &updates, const float threshold) {
                return thresholdEstimate_(updates, threshold).e<int>(0);
            }

            void thresholdEncode(NDArray &updates, NDArray &encoded, float threshold) {
                // we need these blocks in order to know, how many "updates" will be processed by each GPU block
                auto blocks = thresholdEstimate_(updates, threshold);

                const int numThreads = 512;
                const int numBlocks = updates.lengthOf() / numThreads + (updates.lengthOf() % numThreads ? 1 : 0);

                const int prefixThreads = 512;
                int numElts = numBlocks;
                int level = 0;

                // here we just calculate number of sumBlock arrays
                do {
                    int numPrefixBlocks = sd::math::nd4j_max<int>(1, sd::math::nd4j_ceil<float, int>((float) numElts / (2.0f * prefixThreads)));
                    if (numBlocks > 1) {
                        level++;
                    }
                    numElts = numPrefixBlocks;
                } while (numElts > 1);



                std::vector<NDArray> tempArrays(level);
                std::vector<Nd4jPointer> pointers(level);

                level = 0;
                numElts = numBlocks;

                do {
                    int numPrefixBlocks = sd::math::nd4j_max<int>(1, sd::math::nd4j_ceil<float, int>((float) numElts / (2.0f * prefixThreads)));
                    if (numPrefixBlocks > 1) {
                        tempArrays[level] = std::move(NDArrayFactory::create<int>('c', {numPrefixBlocks}));
                        pointers[level] = tempArrays[level++].specialBuffer();
                    }
                    numElts = numPrefixBlocks;
                } while (numElts > 1);

                PointersManager pm(LaunchContext::defaultContext(), "thresholdEncode");
                auto dptr = pm.replicatePointer(pointers.data(), pointers.size() * 8);
                auto offsets = NDArrayFactory::create<int>('c', {numBlocks});

                // we want to check, if we're hiting external limit on number of encoded elements
                auto numMatches = blocks.e<int>(0);
                if (numMatches > encoded.lengthOf() - 4) {
                    blocks.p(0, encoded.lengthOf() - 4);
                    blocks.syncToDevice();
                }

                NDArray::prepareSpecialUse({}, {&encoded, &updates});

                // filling offsets
                encodeThresholdP2Int_(reinterpret_cast<void **>(dptr),
                                      reinterpret_cast<int*>(blocks.specialBuffer()),
                                      numBlocks,
                                      reinterpret_cast<int*>(offsets.specialBuffer()));

                NDArray::registerSpecialUse({&blocks, &offsets}, {});
                pm.synchronize();


                encodeThresholdP3_(updates.specialBuffer(),
                                   updates.shapeInfo(),
                                   reinterpret_cast<int*>(offsets.specialBuffer()),
                                   updates.lengthOf(),
                                   reinterpret_cast<int*>(encoded.specialBuffer()));

                pm.synchronize();

                NDArray::registerSpecialUse({&encoded, &updates}, {});
            }

            void thresholdDecode(const NDArray &encoded, NDArray &updates) {
                dim3 launchDims(128, 512, 512);
                auto xType = updates.dataType();

                NDArray::prepareSpecialUse({&updates}, {&encoded});
                BUILD_SINGLE_SELECTOR(xType, decoderKernelGeneric, (launchDims, LaunchContext::defaultContext()->getCudaStream(), encoded.specialBuffer(), updates.lengthOf(), updates.specialBuffer()), FLOAT_TYPES);
                NDArray::registerSpecialUse({&updates}, {&encoded});
            }
        }
    }
}
