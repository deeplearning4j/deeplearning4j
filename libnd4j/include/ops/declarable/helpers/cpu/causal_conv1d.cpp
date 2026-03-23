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
                           NDArray* stateIn, NDArray* output, NDArray* stateOut, int activation) {
    const auto B = x->sizeAt(0);
    const auto L = x->sizeAt(1);
    const auto D = x->sizeAt(2);
    const auto K = weight->sizeAt(1);

    const T* xBuf = x->bufferAsT<T>();
    const T* wBuf = weight->bufferAsT<T>();
    const T* bBuf = bias != nullptr ? bias->bufferAsT<T>() : nullptr;
    const T* sInBuf = stateIn != nullptr ? stateIn->bufferAsT<T>() : nullptr;
    T* outBuf = output->bufferAsT<T>();
    T* sOutBuf = stateOut->bufferAsT<T>();

    const auto xS0 = x->strideAt(0), xS1 = x->strideAt(1), xS2 = x->strideAt(2);
    const auto wS0 = weight->strideAt(0), wS1 = weight->strideAt(1);
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

            // Causal convolution: output[t] = sum_{kk=0}^{K-1} weight[d,kk] * x[t-kk]
            for (LongType t = 0; t < L; ++t) {
                T sum = static_cast<T>(0);
                for (LongType kk = 0; kk < K; ++kk) {
                    LongType srcT = t - kk;
                    T x_val;
                    if (srcT >= 0) {
                        x_val = xBuf[b * xS0 + srcT * xS1 + d * xS2];
                    } else if (sInBuf != nullptr) {
                        LongType stateIdx = (K - 1) + srcT;
                        x_val = (stateIdx >= 0) ? sInBuf[b * siS0 + d * siS1 + stateIdx * siS2] : static_cast<T>(0);
                    } else {
                        x_val = static_cast<T>(0);
                    }
                    sum += wBuf[d * wS0 + kk * wS1] * x_val;
                }

                if (bBuf != nullptr) sum += bBuf[d];

                // SiLU activation: x * sigmoid(x)
                if (activation == 1) {
                    T sig = static_cast<T>(1) / (static_cast<T>(1) + sd::math::sd_exp<T, T>(-sum));
                    sum = sum * sig;
                }

                outBuf[b * oS0 + t * oS1 + d * oS2] = sum;
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
                   NDArray* stateIn, NDArray* output, NDArray* stateOut, int activation) {
    NDArray::preparePrimaryUse({output, stateOut}, {x, weight, bias});
    if (stateIn != nullptr) NDArray::preparePrimaryUse({}, {stateIn});

    BUILD_SINGLE_SELECTOR(x->dataType(), causalConv1d_,
        (context, x, weight, bias, stateIn, output, stateOut, activation), SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output, stateOut}, {x, weight, bias});
    if (stateIn != nullptr) NDArray::registerPrimaryUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
