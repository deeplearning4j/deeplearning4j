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

#include <execution/Threads.h>
#include <ops/declarable/helpers/rwkv_wkv.h>
#include <system/type_boilerplate.h>

#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// Layout: sequence tensors [B, T, H, S] flat 'c'; tf [H, S]; state [B, H, S, S].
template <typename T>
static void rwkvWkv6_(NDArray* kA, NDArray* vA, NDArray* rA, NDArray* tfA, NDArray* tdA,
                      NDArray* stateA, NDArray* outA) {
    const LongType B = kA->sizeAt(0), Tn = kA->sizeAt(1), H = kA->sizeAt(2), S = kA->sizeAt(3);
    const T* k = kA->bufferAsT<T>();
    const T* v = vA->bufferAsT<T>();
    const T* r = rA->bufferAsT<T>();
    const T* tf = tfA->bufferAsT<T>();
    const T* td = tdA->bufferAsT<T>();
    const T* state0 = stateA->bufferAsT<T>();
    T* out = outA->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto bh = start; bh < stop; bh++) {
            const LongType b = bh / H, h = bh % H;
            std::vector<T> ss(S * S);
            for (LongType i = 0; i < S * S; i++) ss[i] = state0[(b * H + h) * S * S + i];
            std::vector<T> y(S);

            for (LongType t = 0; t < Tn; t++) {
                const LongType base = ((b * Tn + t) * H + h) * S;
                for (LongType j = 0; j < S; j++) y[j] = static_cast<T>(0);
                for (LongType i = 0; i < S; i++) {
                    const T ki = k[base + i], ri = r[base + i];
                    const T tfi = tf[h * S + i], tdi = td[base + i];
                    T* ssRow = ss.data() + i * S;
                    for (LongType j = 0; j < S; j++) {
                        const T kv = ki * v[base + j];
                        y[j] += ri * (tfi * kv + ssRow[j]);
                        ssRow[j] = tdi * ssRow[j] + kv;
                    }
                }
                for (LongType j = 0; j < S; j++) out[base + j] = y[j];
            }
        }
    };
    samediff::Threads::parallel_for(func, 0, B * H);
}

template <typename T>
static void rwkvWkv7_(NDArray* rA, NDArray* wA, NDArray* kA, NDArray* vA, NDArray* aA,
                      NDArray* bA, NDArray* stateA, NDArray* outA) {
    const LongType B = kA->sizeAt(0), Tn = kA->sizeAt(1), H = kA->sizeAt(2), S = kA->sizeAt(3);
    const T* r = rA->bufferAsT<T>();
    const T* w = wA->bufferAsT<T>();
    const T* k = kA->bufferAsT<T>();
    const T* v = vA->bufferAsT<T>();
    const T* a = aA->bufferAsT<T>();
    const T* bb = bA->bufferAsT<T>();
    const T* state0 = stateA->bufferAsT<T>();
    T* out = outA->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto bh = start; bh < stop; bh++) {
            const LongType b = bh / H, h = bh % H;
            std::vector<T> ss(S * S);
            for (LongType i = 0; i < S * S; i++) ss[i] = state0[(b * H + h) * S * S + i];
            std::vector<T> sa(S), y(S);

            for (LongType t = 0; t < Tn; t++) {
                const LongType base = ((b * Tn + t) * H + h) * S;
                // sa[j] = sum_i a[i]*state[i,j]
                for (LongType j = 0; j < S; j++) sa[j] = static_cast<T>(0);
                for (LongType i = 0; i < S; i++) {
                    const T ai = a[base + i];
                    const T* ssRow = ss.data() + i * S;
                    for (LongType j = 0; j < S; j++) sa[j] += ai * ssRow[j];
                }
                // state update + output from the updated state
                for (LongType j = 0; j < S; j++) y[j] = static_cast<T>(0);
                for (LongType i = 0; i < S; i++) {
                    const T ki = k[base + i], wi = w[base + i], bi = bb[base + i], ri = r[base + i];
                    T* ssRow = ss.data() + i * S;
                    for (LongType j = 0; j < S; j++) {
                        ssRow[j] = wi * ssRow[j] + ki * v[base + j] + bi * sa[j];
                        y[j] += ri * ssRow[j];
                    }
                }
                for (LongType j = 0; j < S; j++) out[base + j] = y[j];
            }
        }
    };
    samediff::Threads::parallel_for(func, 0, B * H);
}

void rwkvWkv6(LaunchContext* context, NDArray* k, NDArray* v, NDArray* r, NDArray* tf, NDArray* td,
              NDArray* state, NDArray* output) {
    BUILD_SINGLE_SELECTOR(k->dataType(), rwkvWkv6_, (k, v, r, tf, td, state, output), SD_FLOAT_TYPES);
}

void rwkvWkv7(LaunchContext* context, NDArray* r, NDArray* w, NDArray* k, NDArray* v, NDArray* a,
              NDArray* b, NDArray* state, NDArray* output) {
    BUILD_SINGLE_SELECTOR(k->dataType(), rwkvWkv7_, (r, w, k, v, a, b, state, output), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
