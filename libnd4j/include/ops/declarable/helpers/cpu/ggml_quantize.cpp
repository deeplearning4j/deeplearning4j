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

#include <ops/declarable/helpers/ggml_quantize.h>
#include <system/openmp_pragmas.h>
#include <types/float16.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int QK = 32;         // block size for Q4_0 and Q8_0
static constexpr int Q4_0_BYTES = 18; // 2 (fp16 d) + 16 (32 nibbles)
static constexpr int Q8_0_BYTES = 34; // 2 (fp16 d) + 32 (int8)

static SD_INLINE void writeFp16(uint8_t* dst, float v) {
    float16 h = v;
    std::memcpy(dst, &h, 2);
}

// Q4_0: d = max/-8; qs[j] packs elements 2j (low nibble) and 2j+1 (high nibble)
// — adjacent packing matching dequantize_q4_0 in ggml_dequantize.cpp.
static void quantizeQ4_0(const float* x, uint8_t* out, LongType numElements) {
    const LongType nb = numElements / QK;
    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < nb; b++) {
        const float* xb = x + b * QK;
        uint8_t* ob = out + b * Q4_0_BYTES;

        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < QK; j++) {
            const float v = xb[j];
            const float a = std::fabs(v);
            if (a > amax) { amax = a; max = v; }
        }
        const float d = max / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        writeFp16(ob, d);
        uint8_t* qs = ob + 2;
        for (int j = 0; j < QK / 2; j++) {
            const float x0 = xb[j * 2] * id;
            const float x1 = xb[j * 2 + 1] * id;
            int xi0 = static_cast<int>(x0 + 8.5f);
            int xi1 = static_cast<int>(x1 + 8.5f);
            if (xi0 < 0) xi0 = 0; else if (xi0 > 15) xi0 = 15;
            if (xi1 < 0) xi1 = 0; else if (xi1 > 15) xi1 = 15;
            qs[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
        }
    }
}

// Q8_0: d = max|x|/127; qs[j] = round(x[j]/d) as int8.
static void quantizeQ8_0(const float* x, uint8_t* out, LongType numElements) {
    const LongType nb = numElements / QK;
    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < nb; b++) {
        const float* xb = x + b * QK;
        uint8_t* ob = out + b * Q8_0_BYTES;

        float amax = 0.0f;
        for (int j = 0; j < QK; j++) amax = std::max(amax, std::fabs(xb[j]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        writeFp16(ob, d);
        int8_t* qs = reinterpret_cast<int8_t*>(ob + 2);
        for (int j = 0; j < QK; j++) {
            int q = static_cast<int>(std::lround(xb[j] * id));
            if (q > 127) q = 127; else if (q < -128) q = -128;
            qs[j] = static_cast<int8_t>(q);
        }
    }
}

void ggmlQuantize(LaunchContext* context, NDArray* input, NDArray* output, int quantType) {
    const auto* x = input->bufferAsT<float>();
    auto* out = output->bufferAsT<uint8_t>();
    const LongType n = input->lengthOf();

    switch (quantType) {
        case 0: quantizeQ4_0(x, out, n); break;  // GGML_QUANT_Q4_0
        case 4: quantizeQ8_0(x, out, n); break;  // GGML_QUANT_Q8_0
        default: THROW_EXCEPTION("ggmlQuantize: unsupported quantType (only Q4_0=0 and Q8_0=4 supported)");
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
