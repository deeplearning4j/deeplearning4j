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
#include <ops/declarable/helpers/gated_linear_attn.h>
#include <system/type_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// One thread per (b,h). g may be null (no decay). ss is a per-(b,h) working slice.
template <typename T>
SD_KERNEL void glaKernel(const T* q, const T* k, const T* v, const T* g, T* ss, T* out,
                         LongType B, LongType Tn, LongType H, LongType S, T scale) {
    const LongType bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H) return;
    const LongType b = bh / H, h = bh % H;
    T* st = ss + bh * S * S;

    for (LongType t = 0; t < Tn; t++) {
        const LongType base = ((b * Tn + t) * H + h) * S;
        for (LongType j = 0; j < S; j++) {
            T oj = static_cast<T>(0);
            for (LongType i = 0; i < S; i++) {
                const T gi = g != nullptr ? g[base + i] : static_cast<T>(1);
                st[i * S + j] = gi * st[i * S + j] + k[base + i] * v[base + j];
                oj += q[base + i] * st[i * S + j];
            }
            out[base + j] = oj * scale;
        }
    }
}

template <typename T>
static void glaLaunch(LaunchContext* context, NDArray* q, NDArray* k, NDArray* v, NDArray* gate,
                      NDArray* scratch, NDArray* output, double scaleD) {
    const LongType B = q->sizeAt(0), Tn = q->sizeAt(1), H = q->sizeAt(2), S = q->sizeAt(3);
    auto* stream = context->getCudaStream();
    const int threads = 128;
    const int grid = static_cast<int>((B * H + threads - 1) / threads);
    const T* g = gate != nullptr ? reinterpret_cast<const T*>(gate->specialBuffer()) : nullptr;
    glaKernel<T><<<grid, threads, 0, *stream>>>(
        reinterpret_cast<const T*>(q->specialBuffer()), reinterpret_cast<const T*>(k->specialBuffer()),
        reinterpret_cast<const T*>(v->specialBuffer()), g,
        reinterpret_cast<T*>(scratch->specialBuffer()), reinterpret_cast<T*>(output->specialBuffer()),
        B, Tn, H, S, static_cast<T>(scaleD));
}

void gatedLinearAttn(LaunchContext* context, NDArray* q, NDArray* k, NDArray* v, NDArray* gate,
                     NDArray* state, NDArray* output, double scale) {
    std::vector<LongType> stShape(state->rankOf());
    for (int i = 0; i < state->rankOf(); i++) stShape[i] = state->sizeAt(i);
    NDArray scratch('c', stShape, state->dataType(), context);
    scratch.assign(state);

    if (gate != nullptr) {
        NDArray::prepareSpecialUse({output, &scratch}, {q, k, v, gate});
    } else {
        NDArray::prepareSpecialUse({output, &scratch}, {q, k, v});
    }
    BUILD_SINGLE_SELECTOR(q->dataType(), glaLaunch, (context, q, k, v, gate, &scratch, output, scale),
                          SD_FLOAT_TYPES);
    DebugHelper::checkGlobalErrorCode("gated_linear_attn kernel failed");
    if (gate != nullptr) {
        NDArray::registerSpecialUse({output, &scratch}, {q, k, v, gate});
    } else {
        NDArray::registerSpecialUse({output, &scratch}, {q, k, v});
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
