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
// Grammar/constrained decoding token bitmask — CPU implementation.
// Applies packed uint32 bitmask to logits: bit=0 → logit = -inf.
//

#include <execution/Threads.h>
#include <ops/declarable/helpers/apply_token_bitmask.h>

#include <cmath>
#include <limits>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void applyTokenBitmaskCpu_(NDArray* logits, NDArray* bitmask, NDArray* indices) {
    const LongType batchSize = logits->sizeAt(0);
    const LongType vocabSize = logits->sizeAt(1);
    const LongType bitmaskWords = bitmask->sizeAt(1);  // ceil(vocabSize / 32)

    T* logitsPtr = logits->bufferAsT<T>();
    const uint32_t* maskPtr = reinterpret_cast<const uint32_t*>(bitmask->buffer());
    const int32_t* idxPtr = (indices != nullptr) ?
        reinterpret_cast<const int32_t*>(indices->buffer()) : nullptr;

    const T negInf = static_cast<T>(-std::numeric_limits<float>::infinity());

    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; ++b) {
            // Determine which bitmask row to use
            LongType maskRow = b;
            if (idxPtr != nullptr) {
                maskRow = static_cast<LongType>(idxPtr[b]);
            }

            T* batchLogits = logitsPtr + b * vocabSize;
            const uint32_t* batchMask = maskPtr + maskRow * bitmaskWords;

            for (LongType v = 0; v < vocabSize; ++v) {
                LongType wordIdx = v / 32;
                int bitIdx = static_cast<int>(v % 32);

                uint32_t word = batchMask[wordIdx];
                bool allowed = (word >> bitIdx) & 1u;

                if (!allowed) {
                    batchLogits[v] = negInf;
                }
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, batchSize);
}

//////////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////////
void applyTokenBitmaskInplace(LaunchContext* context,
                               NDArray* logits,
                               NDArray* bitmask,
                               NDArray* indices) {
    BUILD_SINGLE_SELECTOR(logits->dataType(), applyTokenBitmaskCpu_,
                          (logits, bitmask, indices), SD_FLOAT_TYPES);

    logits->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
