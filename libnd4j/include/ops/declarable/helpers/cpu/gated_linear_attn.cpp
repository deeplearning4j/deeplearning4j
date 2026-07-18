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
#include <ops/declarable/helpers/gated_linear_attn.h>
#include <system/type_boilerplate.h>

#include <vector>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void gla_(NDArray* qA, NDArray* kA, NDArray* vA, NDArray* gA, NDArray* stateA,
                 NDArray* outA, double scaleD) {
    const LongType B = qA->sizeAt(0), Tn = qA->sizeAt(1), H = qA->sizeAt(2), S = qA->sizeAt(3);
    const T scale = static_cast<T>(scaleD);
    const T* q = qA->bufferAsT<T>();
    const T* k = kA->bufferAsT<T>();
    const T* v = vA->bufferAsT<T>();
    const T* g = gA != nullptr ? gA->bufferAsT<T>() : nullptr;
    const T* state0 = stateA->bufferAsT<T>();
    T* out = outA->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto bh = start; bh < stop; bh++) {
            const LongType b = bh / H, h = bh % H;
            std::vector<T> ss(S * S);
            for (LongType i = 0; i < S * S; i++) ss[i] = state0[(b * H + h) * S * S + i];
            std::vector<T> o(S);

            for (LongType t = 0; t < Tn; t++) {
                const LongType base = ((b * Tn + t) * H + h) * S;
                for (LongType j = 0; j < S; j++) o[j] = static_cast<T>(0);
                for (LongType i = 0; i < S; i++) {
                    const T gi = g != nullptr ? g[base + i] : static_cast<T>(1);
                    const T ki = k[base + i], qi = q[base + i];
                    T* ssRow = ss.data() + i * S;
                    for (LongType j = 0; j < S; j++) {
                        ssRow[j] = gi * ssRow[j] + ki * v[base + j];
                        o[j] += qi * ssRow[j];
                    }
                }
                for (LongType j = 0; j < S; j++) out[base + j] = o[j] * scale;
            }
        }
    };
    samediff::Threads::parallel_for(func, 0, B * H);
}

void gatedLinearAttn(LaunchContext* context, NDArray* q, NDArray* k, NDArray* v, NDArray* gate,
                     NDArray* state, NDArray* output, double scale) {
    BUILD_SINGLE_SELECTOR(q->dataType(), gla_, (q, k, v, gate, state, output, scale), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
