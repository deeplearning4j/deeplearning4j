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

#include <helpers/DebugHelper.h>
#include <ops/declarable/helpers/ggml_quantize.h>

#include <cuda_fp16.h>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int QK = 32;
static constexpr int Q4_0_BYTES = 18;
static constexpr int Q8_0_BYTES = 34;

// One thread per 32-element block.
SD_KERNEL void quantizeQ4_0Kernel(const float* __restrict__ x, uint8_t* __restrict__ out, LongType numBlocks) {
    const LongType b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= numBlocks) return;

    const float* xb = x + b * QK;
    uint8_t* ob = out + b * Q4_0_BYTES;

    float amax = 0.0f, maxValue = 0.0f;
    for (int j = 0; j < QK; j++) {
        const float v = xb[j];
        const float a = fabsf(v);
        if (a > amax) { amax = a; maxValue = v; }
    }
    const float d = maxValue / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    const half hd = __float2half(d);
    reinterpret_cast<half*>(ob)[0] = hd;
    uint8_t* qs = ob + 2;
    for (int j = 0; j < QK / 2; j++) {
        const float x0 = xb[j * 2] * id;
        const float x1 = xb[j * 2 + 1] * id;
        int xi0 = static_cast<int>(x0 + 8.5f);
        int xi1 = static_cast<int>(x1 + 8.5f);
        xi0 = min(15, max(0, xi0));
        xi1 = min(15, max(0, xi1));
        qs[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
    }
}

SD_KERNEL void quantizeQ8_0Kernel(const float* __restrict__ x, uint8_t* __restrict__ out, LongType numBlocks) {
    const LongType b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= numBlocks) return;

    const float* xb = x + b * QK;
    uint8_t* ob = out + b * Q8_0_BYTES;

    float amax = 0.0f;
    for (int j = 0; j < QK; j++) amax = fmaxf(amax, fabsf(xb[j]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    const half hd = __float2half(d);
    reinterpret_cast<half*>(ob)[0] = hd;
    int8_t* qs = reinterpret_cast<int8_t*>(ob + 2);
    for (int j = 0; j < QK; j++) {
        int q = __float2int_rn(xb[j] * id);
        q = min(127, max(-128, q));
        qs[j] = static_cast<int8_t>(q);
    }
}

void ggmlQuantize(LaunchContext* context, NDArray* input, NDArray* output, int quantType) {
    NDArray::prepareSpecialUse({output}, {input});

    const auto* x = reinterpret_cast<const float*>(input->specialBuffer());
    auto* out = reinterpret_cast<uint8_t*>(output->specialBuffer());
    const LongType n = input->lengthOf();
    const LongType numBlocks = n / QK;

    auto* stream = context->getCudaStream();
    const int threads = 128;
    const int grid = static_cast<int>((numBlocks + threads - 1) / threads);

    switch (quantType) {
        case 0: quantizeQ4_0Kernel<<<grid, threads, 0, *stream>>>(x, out, numBlocks); break;
        case 4: quantizeQ8_0Kernel<<<grid, threads, 0, *stream>>>(x, out, numBlocks); break;
        default: THROW_EXCEPTION("ggmlQuantize: unsupported quantType (only Q4_0=0 and Q8_0=4 supported)");
    }
    DebugHelper::checkGlobalErrorCode("ggml_quantize kernel failed");

    NDArray::registerSpecialUse({output}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
