/* ******************************************************************************
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
// Fused RMS Normalization CPU implementation
// Computes: output = input / sqrt(mean(input^2) + epsilon) * gamma
// 2-pass fused implementation with parallel_tad for multi-threaded execution.
//

#include <execution/Threads.h>
#include <ops/declarable/helpers/rms_norm.h>
#if NOT_EXCLUDED(OP_rms_norm)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void rmsNormCpu_(NDArray* input, NDArray* gamma, NDArray* output, float epsilon) {
    const LongType numRows = input->lengthOf() / input->sizeAt(-1);
    const LongType rowLen = input->sizeAt(-1);

    const T* x = input->bufferAsT<T>();
    T* z = output->bufferAsT<T>();
    const T* g = gamma != nullptr ? gamma->bufferAsT<T>() : nullptr;
    const T eps = static_cast<T>(epsilon);

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * rowLen;
            T* zRow = z + row * rowLen;

            // Pass 1: compute sum of squares
            T sumSq = static_cast<T>(0);
            for (LongType i = 0; i < rowLen; ++i) {
                sumSq += xRow[i] * xRow[i];
            }
            const T invRms = static_cast<T>(1) / sd::math::sd_sqrt<T, T>(sumSq / static_cast<T>(rowLen) + eps);

            // Pass 2: normalize and scale
            if (g != nullptr) {
                for (LongType i = 0; i < rowLen; ++i) {
                    zRow[i] = xRow[i] * invRms * g[i];
                }
            } else {
                for (LongType i = 0; i < rowLen; ++i) {
                    zRow[i] = xRow[i] * invRms;
                }
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
}

void rmsNormCpu(NDArray* input, NDArray* gamma, NDArray* output, float epsilon) {
    input->syncToHost();
    if (gamma != nullptr) {
        gamma->syncToHost();
    }

    BUILD_SINGLE_SELECTOR(input->dataType(), rmsNormCpu_, (input, gamma, output, epsilon), SD_FLOAT_TYPES);

    output->tickWriteHost();
    output->syncToDevice();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
