/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//

#include <cuda_runtime.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/ctc_decode.h>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_ctc_greedy_decoder)

namespace sd {
namespace ops {
namespace helpers {

template <typename T, typename I>
SD_KERNEL static void ctcGreedyDecodeKernel(
    const T* logits,
    const I* sequenceLength,
    sd::LongType* decoded,
    T* logProb,
    const sd::LongType batchSize,
    const sd::LongType maxTimeSteps,
    const sd::LongType numClasses,
    const sd::LongType blankIndex,
    const bool mergeRepeated,
    const sd::LongType logitsBatchStride,
    const sd::LongType logitsTimeStride,
    const sd::LongType logitsClassStride,
    const sd::LongType decodedBatchStride,
    const sd::LongType decodedTimeStride) {

    // Each block handles one batch element
    const sd::LongType b = blockIdx.x;
    if (b >= batchSize) return;

    // Get sequence length for this batch
    sd::LongType seqLen = maxTimeSteps;
    if (sequenceLength != nullptr) {
        seqLen = static_cast<sd::LongType>(sequenceLength[b]);
        if (seqLen > maxTimeSteps) seqLen = maxTimeSteps;
    }

    const T* batchLogits = logits + b * logitsBatchStride;
    sd::LongType* batchDecoded = decoded + b * decodedBatchStride;

    // Shared memory for reduction (argmax)
    extern __shared__ char sharedMem[];
    T* sharedVals = reinterpret_cast<T*>(sharedMem);
    sd::LongType* sharedIdx = reinterpret_cast<sd::LongType*>(sharedMem + blockDim.x * sizeof(T));

    // Thread 0 initializes decoded output to -1
    if (threadIdx.x == 0) {
        for (sd::LongType t = 0; t < maxTimeSteps; t++) {
            batchDecoded[t * decodedTimeStride] = -1;
        }
    }
    __syncthreads();

    T totalLogProb = static_cast<T>(0);
    sd::LongType prevToken = -1;
    sd::LongType decodedIdx = 0;

    for (sd::LongType t = 0; t < seqLen; t++) {
        const T* timeLogits = batchLogits + t * logitsTimeStride;

        // Parallel argmax using all threads in block
        T localMax = -1e38f;
        sd::LongType localIdx = 0;

        for (sd::LongType c = threadIdx.x; c < numClasses; c += blockDim.x) {
            T val = timeLogits[c * logitsClassStride];
            if (val > localMax) {
                localMax = val;
                localIdx = c;
            }
        }

        sharedVals[threadIdx.x] = localMax;
        sharedIdx[threadIdx.x] = localIdx;
        __syncthreads();

        // Reduce to find global max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                if (sharedVals[threadIdx.x + stride] > sharedVals[threadIdx.x]) {
                    sharedVals[threadIdx.x] = sharedVals[threadIdx.x + stride];
                    sharedIdx[threadIdx.x] = sharedIdx[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        // Thread 0 applies CTC decoding rules
        if (threadIdx.x == 0) {
            sd::LongType maxIdx = sharedIdx[0];
            T maxVal = sharedVals[0];

            totalLogProb += maxVal;

            bool shouldOutput = false;
            if (maxIdx != blankIndex) {
                if (mergeRepeated) {
                    if (maxIdx != prevToken) {
                        shouldOutput = true;
                    }
                } else {
                    shouldOutput = true;
                }
            }

            if (shouldOutput) {
                batchDecoded[decodedIdx * decodedTimeStride] = maxIdx;
                decodedIdx++;
                prevToken = maxIdx;
            } else if (maxIdx == blankIndex) {
                prevToken = -1;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes log probability
    if (threadIdx.x == 0) {
        logProb[b] = totalLogProb;
    }
}

template <typename T, typename I>
static void ctcGreedyDecodeCuda_(sd::LaunchContext* context,
                                  NDArray* logits,
                                  NDArray* sequenceLength,
                                  NDArray* decoded,
                                  NDArray* logProb,
                                  sd::LongType blankIndex,
                                  bool mergeRepeated) {

    const auto batchSize = logits->sizeAt(0);
    const auto maxTimeSteps = logits->sizeAt(1);
    const auto numClasses = logits->sizeAt(2);

    const auto logitsBatchStride = logits->strideAt(0);
    const auto logitsTimeStride = logits->strideAt(1);
    const auto logitsClassStride = logits->strideAt(2);

    const auto decodedBatchStride = decoded->strideAt(0);
    const auto decodedTimeStride = decoded->strideAt(1);

    // Determine block size based on number of classes
    int blockSize = 256;
    if (numClasses < 256) blockSize = 128;
    if (numClasses < 128) blockSize = 64;

    // Shared memory size
    size_t sharedMemSize = blockSize * (sizeof(T) + sizeof(sd::LongType));

    dim3 grid(batchSize);
    dim3 block(blockSize);

    PointersManager manager(context, "ctcGreedyDecode");

    const T* logitsBuffer = logits->specialBufferasT<T>();
    const I* seqLenBuffer = sequenceLength != nullptr ? sequenceLength->specialBufferasT<I>() : nullptr;
    sd::LongType* decodedBuffer = decoded->specialBufferasT<sd::LongType>();
    T* logProbBuffer = logProb->specialBufferasT<T>();

    ctcGreedyDecodeKernel<T, I><<<grid, block, sharedMemSize, *context->getCudaStream()>>>(
        logitsBuffer,
        seqLenBuffer,
        decodedBuffer,
        logProbBuffer,
        batchSize,
        maxTimeSteps,
        numClasses,
        blankIndex,
        mergeRepeated,
        logitsBatchStride,
        logitsTimeStride,
        logitsClassStride,
        decodedBatchStride,
        decodedTimeStride);

    manager.synchronize();
}

void ctcGreedyDecode(sd::LaunchContext* context,
                      NDArray* logits,
                      NDArray* sequenceLength,
                      NDArray* decoded,
                      NDArray* logProb,
                      int blankIndex,
                      bool mergeRepeated) {

    const auto numClasses = logits->sizeAt(2);
    sd::LongType actualBlankIndex = blankIndex < 0 ? numClasses - 1 : blankIndex;

    auto xType = logits->dataType();
    auto iType = sequenceLength != nullptr ? sequenceLength->dataType() : sd::DataType::INT32;

    BUILD_DOUBLE_SELECTOR(xType, iType, ctcGreedyDecodeCuda_,
                          (context, logits, sequenceLength, decoded, logProb, actualBlankIndex, mergeRepeated),
                          SD_FLOAT_TYPES, SD_INDEXING_TYPES);
}

BUILD_DOUBLE_TEMPLATE(void ctcGreedyDecodeCuda_,
                      (sd::LaunchContext* context, NDArray* logits, NDArray* sequenceLength,
                       NDArray* decoded, NDArray* logProb, sd::LongType blankIndex, bool mergeRepeated),
                      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
