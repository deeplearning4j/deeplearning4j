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
// Grammar/constrained decoding bitmask CUDA kernel.
// Applies a packed uint32 bitmask to logits in-place: disallowed tokens -> -inf.
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <types/float16.h>
#include <ops/declarable/helpers/apply_token_bitmask.h>
#include <math/templatemath.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Bitmask kernel: each thread handles 32 token positions (one uint32 word)
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void applyTokenBitmaskKernel(
    T* __restrict__ logits,
    const uint32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    const LongType batchSize,
    const LongType vocabSize,
    const LongType bitmaskWordsPerRow) {

    // blockIdx.x = batch index, threads handle bitmask words
    const LongType batchIdx = blockIdx.x;
    if (batchIdx >= batchSize) return;

    // Resolve which bitmask row to use
    const LongType maskRow = (indices != nullptr) ? static_cast<LongType>(indices[batchIdx]) : batchIdx;

    T* batchLogits = logits + batchIdx * vocabSize;
    const uint32_t* batchMask = bitmask + maskRow * bitmaskWordsPerRow;

    const T negInf = static_cast<T>(-1e30f);

    for (LongType wordIdx = threadIdx.x; wordIdx < bitmaskWordsPerRow; wordIdx += blockDim.x) {
        uint32_t maskWord = batchMask[wordIdx];

        // If all bits set, all tokens allowed -- skip
        if (maskWord == 0xFFFFFFFF) continue;

        // If no bits set, all tokens disallowed -- bulk set
        LongType tokenBase = wordIdx * 32;
        if (maskWord == 0) {
            for (int bit = 0; bit < 32 && (tokenBase + bit) < vocabSize; bit++) {
                batchLogits[tokenBase + bit] = negInf;
            }
            continue;
        }

        // Sparse case: check each bit
        for (int bit = 0; bit < 32; bit++) {
            LongType tokenPos = tokenBase + bit;
            if (tokenPos >= vocabSize) break;

            if ((maskWord & (1u << bit)) == 0) {
                // Bit is 0 = disallowed
                batchLogits[tokenPos] = negInf;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Vectorized variant using float4 for coalesced memory access
//////////////////////////////////////////////////////////////////////////////
SD_KERNEL void applyTokenBitmaskFloat4Kernel(
    float* __restrict__ logits,
    const uint32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    const LongType batchSize,
    const LongType vocabSize,
    const LongType bitmaskWordsPerRow) {

    const LongType batchIdx = blockIdx.x;
    if (batchIdx >= batchSize) return;

    const LongType maskRow = (indices != nullptr) ? static_cast<LongType>(indices[batchIdx]) : batchIdx;

    float* batchLogits = logits + batchIdx * vocabSize;
    const uint32_t* batchMask = bitmask + maskRow * bitmaskWordsPerRow;

    const float negInf = -1e30f;

    for (LongType wordIdx = threadIdx.x; wordIdx < bitmaskWordsPerRow; wordIdx += blockDim.x) {
        uint32_t maskWord = batchMask[wordIdx];

        if (maskWord == 0xFFFFFFFF) continue;

        LongType tokenBase = wordIdx * 32;

        // Process 4 floats at a time (128 bits = float4)
        for (int group = 0; group < 8; group++) {
            LongType groupBase = tokenBase + group * 4;
            if (groupBase + 3 >= vocabSize) {
                // Handle tail elements individually
                for (int bit = group * 4; bit < 32 && (tokenBase + bit) < vocabSize; bit++) {
                    if ((maskWord & (1u << bit)) == 0) {
                        batchLogits[tokenBase + bit] = negInf;
                    }
                }
                break;
            }

            // Check if any of the 4 bits are cleared
            uint32_t groupBits = (maskWord >> (group * 4)) & 0xF;
            if (groupBits == 0xF) continue;  // All 4 allowed

            // Load, mask, store
            float4* ptr = reinterpret_cast<float4*>(batchLogits + groupBase);
            float4 vals = *ptr;

            if ((groupBits & 1) == 0) vals.x = negInf;
            if ((groupBits & 2) == 0) vals.y = negInf;
            if ((groupBits & 4) == 0) vals.z = negInf;
            if ((groupBits & 8) == 0) vals.w = negInf;

            *ptr = vals;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////////
void applyTokenBitmaskInplace(LaunchContext* context,
                               NDArray* logits,
                               NDArray* bitmask,
                               NDArray* indices) {
    const LongType batchSize = logits->sizeAt(0);
    const LongType vocabSize = logits->sizeAt(1);
    const LongType bitmaskWordsPerRow = (vocabSize + 31) / 32;

    NDArray::prepareSpecialUse({logits}, {bitmask});
    if (indices != nullptr) {
        NDArray::prepareSpecialUse({}, {indices});
    }

    auto stream = context->getCudaStream();
    auto dtype = logits->dataType();

    // 256 threads, each handles one uint32 word (32 tokens)
    int threads = 256;
    if (bitmaskWordsPerRow < 256) {
        threads = ((bitmaskWordsPerRow + 31) / 32) * 32;
        if (threads < 32) threads = 32;
    }

    const int32_t* indicesPtr = (indices != nullptr) ?
        reinterpret_cast<const int32_t*>(indices->specialBuffer()) : nullptr;

    if (dtype == DataType::FLOAT32 && (vocabSize % 4 == 0)) {
        // Use vectorized float4 kernel for aligned float32
        applyTokenBitmaskFloat4Kernel<<<batchSize, threads, 0, *stream>>>(
            reinterpret_cast<float*>(logits->specialBuffer()),
            reinterpret_cast<const uint32_t*>(bitmask->specialBuffer()),
            indicesPtr,
            batchSize, vocabSize, bitmaskWordsPerRow);
    } else if (dtype == DataType::FLOAT32) {
        applyTokenBitmaskKernel<float><<<batchSize, threads, 0, *stream>>>(
            reinterpret_cast<float*>(logits->specialBuffer()),
            reinterpret_cast<const uint32_t*>(bitmask->specialBuffer()),
            indicesPtr,
            batchSize, vocabSize, bitmaskWordsPerRow);
    } else if (dtype == DataType::HALF) {
        applyTokenBitmaskKernel<float16><<<batchSize, threads, 0, *stream>>>(
            reinterpret_cast<float16*>(logits->specialBuffer()),
            reinterpret_cast<const uint32_t*>(bitmask->specialBuffer()),
            indicesPtr,
            batchSize, vocabSize, bitmaskWordsPerRow);
    } else {
        THROW_EXCEPTION("applyTokenBitmaskInplace: unsupported data type (need FLOAT32 or HALF)");
    }

    DebugHelper::checkGlobalErrorCode("applyTokenBitmaskInplace failed");
    NDArray::registerSpecialUse({logits}, {bitmask});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
