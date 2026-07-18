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
#include <ops/declarable/helpers/rwkv_wkv.h>
#include <system/type_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// One thread per (b,h). ss is a per-(b,h) working state slice [S*S] in global scratch.
template <typename T>
SD_KERNEL void rwkvWkv6Kernel(const T* k, const T* v, const T* r, const T* tf, const T* td,
                              T* ss, T* out, LongType B, LongType Tn, LongType H, LongType S) {
    const LongType bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H) return;
    const LongType b = bh / H, h = bh % H;
    T* st = ss + bh * S * S;

    for (LongType t = 0; t < Tn; t++) {
        const LongType base = ((b * Tn + t) * H + h) * S;
        for (LongType j = 0; j < S; j++) {
            T yj = static_cast<T>(0);
            for (LongType i = 0; i < S; i++) {
                const T kv = k[base + i] * v[base + j];
                yj += r[base + i] * (tf[h * S + i] * kv + st[i * S + j]);
            }
            out[base + j] = yj;
        }
        for (LongType i = 0; i < S; i++) {
            const T ki = k[base + i], tdi = td[base + i];
            for (LongType j = 0; j < S; j++)
                st[i * S + j] = tdi * st[i * S + j] + ki * v[base + j];
        }
    }
}

template <typename T>
SD_KERNEL void rwkvWkv7Kernel(const T* r, const T* w, const T* k, const T* v, const T* a,
                              const T* bb, T* ss, T* out, LongType B, LongType Tn, LongType H, LongType S) {
    const LongType bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H) return;
    const LongType b = bh / H, h = bh % H;
    T* st = ss + bh * S * S;

    for (LongType t = 0; t < Tn; t++) {
        const LongType base = ((b * Tn + t) * H + h) * S;
        for (LongType j = 0; j < S; j++) {
            // sa[j] = sum_i a[i]*st[i,j]
            T sa = static_cast<T>(0);
            for (LongType i = 0; i < S; i++) sa += a[base + i] * st[i * S + j];
            // update column j and accumulate output
            T yj = static_cast<T>(0);
            for (LongType i = 0; i < S; i++) {
                st[i * S + j] = w[base + i] * st[i * S + j] + k[base + i] * v[base + j] + bb[base + i] * sa;
                yj += r[base + i] * st[i * S + j];
            }
            out[base + j] = yj;
        }
    }
}

template <typename T>
static void rwkvWkv6Launch(LaunchContext* context, NDArray* k, NDArray* v, NDArray* r, NDArray* tf,
                           NDArray* td, NDArray* scratch, NDArray* output) {
    const LongType B = k->sizeAt(0), Tn = k->sizeAt(1), H = k->sizeAt(2), S = k->sizeAt(3);
    auto* stream = context->getCudaStream();
    const int threads = 128;
    const int grid = static_cast<int>((B * H + threads - 1) / threads);
    rwkvWkv6Kernel<T><<<grid, threads, 0, *stream>>>(
        reinterpret_cast<const T*>(k->specialBuffer()), reinterpret_cast<const T*>(v->specialBuffer()),
        reinterpret_cast<const T*>(r->specialBuffer()), reinterpret_cast<const T*>(tf->specialBuffer()),
        reinterpret_cast<const T*>(td->specialBuffer()), reinterpret_cast<T*>(scratch->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()), B, Tn, H, S);
}

template <typename T>
static void rwkvWkv7Launch(LaunchContext* context, NDArray* r, NDArray* w, NDArray* k, NDArray* v,
                           NDArray* a, NDArray* b, NDArray* scratch, NDArray* output) {
    const LongType B = k->sizeAt(0), Tn = k->sizeAt(1), H = k->sizeAt(2), S = k->sizeAt(3);
    auto* stream = context->getCudaStream();
    const int threads = 128;
    const int grid = static_cast<int>((B * H + threads - 1) / threads);
    rwkvWkv7Kernel<T><<<grid, threads, 0, *stream>>>(
        reinterpret_cast<const T*>(r->specialBuffer()), reinterpret_cast<const T*>(w->specialBuffer()),
        reinterpret_cast<const T*>(k->specialBuffer()), reinterpret_cast<const T*>(v->specialBuffer()),
        reinterpret_cast<const T*>(a->specialBuffer()), reinterpret_cast<const T*>(b->specialBuffer()),
        reinterpret_cast<T*>(scratch->specialBuffer()), reinterpret_cast<T*>(output->specialBuffer()),
        B, Tn, H, S);
}

void rwkvWkv6(LaunchContext* context, NDArray* k, NDArray* v, NDArray* r, NDArray* tf, NDArray* td,
              NDArray* state, NDArray* output) {
    // working state copy (kernel mutates it in place)
    std::vector<LongType> stShape(state->rankOf());
    for (int i = 0; i < state->rankOf(); i++) stShape[i] = state->sizeAt(i);
    NDArray scratch('c', stShape, state->dataType(), context);
    scratch.assign(state);

    NDArray::prepareSpecialUse({output, &scratch}, {k, v, r, tf, td});
    BUILD_SINGLE_SELECTOR(k->dataType(), rwkvWkv6Launch, (context, k, v, r, tf, td, &scratch, output),
                          SD_FLOAT_TYPES);
    DebugHelper::checkGlobalErrorCode("rwkv_wkv6 kernel failed");
    NDArray::registerSpecialUse({output, &scratch}, {k, v, r, tf, td});
}

void rwkvWkv7(LaunchContext* context, NDArray* r, NDArray* w, NDArray* k, NDArray* v, NDArray* a,
              NDArray* b, NDArray* state, NDArray* output) {
    std::vector<LongType> stShape(state->rankOf());
    for (int i = 0; i < state->rankOf(); i++) stShape[i] = state->sizeAt(i);
    NDArray scratch('c', stShape, state->dataType(), context);
    scratch.assign(state);

    NDArray::prepareSpecialUse({output, &scratch}, {r, w, k, v, a, b});
    BUILD_SINGLE_SELECTOR(k->dataType(), rwkvWkv7Launch, (context, r, w, k, v, a, b, &scratch, output),
                          SD_FLOAT_TYPES);
    DebugHelper::checkGlobalErrorCode("rwkv_wkv7 kernel failed");
    NDArray::registerSpecialUse({output, &scratch}, {r, w, k, v, a, b});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
