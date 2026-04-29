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

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/causal_conv1d.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void causalConv1d_(LaunchContext* context, NDArray* x, NDArray* weight, NDArray* bias,
                           NDArray* stateIn, NDArray* output, NDArray* stateOut, int activation, int wFormat) {
    const auto B = x->sizeAt(0);
    const auto L = x->sizeAt(1);
    const auto D = x->sizeAt(2);
    const auto K = (wFormat == 0) ? weight->sizeAt(1) : weight->sizeAt(0);

    const T* xBuf = x->bufferAsT<T>();
    const T* wBuf = weight->bufferAsT<T>();
    const T* bBuf = bias != nullptr ? bias->bufferAsT<T>() : nullptr;
    const T* sInBuf = stateIn != nullptr ? stateIn->bufferAsT<T>() : nullptr;
    T* outBuf = output->bufferAsT<T>();
    T* sOutBuf = stateOut->bufferAsT<T>();

    const auto xS0 = x->strideAt(0), xS1 = x->strideAt(1), xS2 = x->strideAt(2);
    // wFormat=0 [D,K]: wChanStride=strideAt(0), wDimStride=strideAt(1)
    // wFormat=1 [K,D]: wDimStride=strideAt(0), wChanStride=strideAt(1)
    const LongType wChanStride = (wFormat == 0) ? weight->strideAt(0) : weight->strideAt(1);
    const LongType wDimStride  = (wFormat == 0) ? weight->strideAt(1) : weight->strideAt(0);
    const auto oS0 = output->strideAt(0), oS1 = output->strideAt(1), oS2 = output->strideAt(2);
    const auto soS0 = stateOut->strideAt(0), soS1 = stateOut->strideAt(1), soS2 = stateOut->strideAt(2);

    LongType siS0 = 0, siS1 = 0, siS2 = 0;
    if (stateIn != nullptr) {
        siS0 = stateIn->strideAt(0);
        siS1 = stateIn->strideAt(1);
        siS2 = stateIn->strideAt(2);
    }

    // Parallel over batch * channels
    auto func = PRAGMA_THREADS_FOR {
        for (auto bd = start; bd < stop; ++bd) {
            const LongType b = bd / D;
            const LongType d = bd % D;

            // Causal correlation (PyTorch F.conv1d convention):
            // output[t] = sum_{kk=0}^{K-1} weight[d, K-1-kk] * x[t-kk]
            // weight[K-1] multiplies x[t] (current), weight[0] multiplies x[t-K+1] (oldest)
            for (LongType t = 0; t < L; ++t) {
                // Accumulate convolution in float to avoid FP16 product overflow
                float sum = 0.0f;
                for (LongType kk = 0; kk < K; ++kk) {
                    LongType srcT = t - kk;
                    float x_val;
                    if (srcT >= 0) {
                        x_val = static_cast<float>(xBuf[b * xS0 + srcT * xS1 + d * xS2]);
                    } else if (sInBuf != nullptr) {
                        LongType stateIdx = (K - 1) + srcT;
                        x_val = (stateIdx >= 0) ? static_cast<float>(sInBuf[b * siS0 + d * siS1 + stateIdx * siS2]) : 0.0f;
                    } else {
                        x_val = 0.0f;
                    }
                    sum += static_cast<float>(wBuf[d * wChanStride + (K - 1 - kk) * wDimStride]) * x_val;
                }

                if (bBuf != nullptr) sum += static_cast<float>(bBuf[d]);

                // SiLU activation: x * sigmoid(x) — computed in float to avoid exp overflow
                if (activation == 1) {
                    float sig = 1.0f / (1.0f + std::exp(-sum));
                    sum = sum * sig;
                }

                outBuf[b * oS0 + t * oS1 + d * oS2] = static_cast<T>(sum);
            }

            // Update conv state: last K-1 timesteps of input
            for (LongType kk = 0; kk < K - 1; ++kk) {
                LongType srcT = L - (K - 1) + kk;
                T val = static_cast<T>(0);
                if (srcT >= 0) {
                    val = xBuf[b * xS0 + srcT * xS1 + d * xS2];
                } else if (sInBuf != nullptr) {
                    LongType stateIdx = (K - 1) + srcT;
                    if (stateIdx >= 0) val = sInBuf[b * siS0 + d * siS1 + stateIdx * siS2];
                }
                sOutBuf[b * soS0 + d * soS1 + kk * soS2] = val;
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, B * D);
}

void causalConv1d(LaunchContext* context, NDArray* x, NDArray* weight, NDArray* bias,
                   NDArray* stateIn, NDArray* output, NDArray* stateOut, int activation, int wFormat) {
    NDArray::preparePrimaryUse({output, stateOut}, {x, weight, bias});
    if (stateIn != nullptr) NDArray::preparePrimaryUse({}, {stateIn});

    BUILD_SINGLE_SELECTOR(x->dataType(), causalConv1d_,
        (context, x, weight, bias, stateIn, output, stateOut, activation, wFormat), SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output, stateOut}, {x, weight, bias});
    if (stateIn != nullptr) NDArray::registerPrimaryUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
