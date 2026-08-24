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
#include <ops/op_types.h>
#include <ops/declarable/helpers/causal_conv1d.h>
#include <ops/declarable/helpers/reproducible_math.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename W, typename S>
static void causalConv1d_(LaunchContext* context, NDArray* x, NDArray* weight, NDArray* bias,
                           NDArray* stateIn, NDArray* actualLen, NDArray* output, NDArray* stateOut,
                           int activation, int wFormat) {
    using PromotedT = typename sd::math::promote_type3<X, W, S>::type;
    using AccT = typename simdOps::AggregateType<PromotedT>::type;

    const auto B = x->sizeAt(0);
    const auto L = x->sizeAt(1);
    const auto D = x->sizeAt(2);
    const auto K = (wFormat == 0) ? weight->sizeAt(1) : weight->sizeAt(0);
    LongType effectiveLen = L;
    if (actualLen != nullptr) {
        effectiveLen = actualLen->e<LongType>(0);
        if (effectiveLen < 0) effectiveLen = 0;
        if (effectiveLen > L) effectiveLen = L;
    }

    const X* xBuf = x->bufferAsT<X>();
    const W* wBuf = weight->bufferAsT<W>();
    const X* bBuf = bias != nullptr ? bias->bufferAsT<X>() : nullptr;
    const S* sInBuf = stateIn != nullptr ? stateIn->bufferAsT<S>() : nullptr;
    X* outBuf = output->bufferAsT<X>();
    S* sOutBuf = stateOut->bufferAsT<S>();

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

            // Causal convolution matching PyTorch F.conv1d with left-padding:
            //   F.conv1d(x, w.unsqueeze(1), padding=K-1)[:, :, :L]
            // With left-padding + truncation, weight[K-1] multiplies x[t] (current),
            // weight[0] multiplies x[t-K+1] (oldest). We iterate lag kk from 0..K-1
            // where srcT = t - kk, so kk=0 is current timestep. Map to weight index
            // K-1-kk so weight[K-1] hits current input.
            for (LongType t = 0; t < L; ++t) {
                AccT sum = static_cast<AccT>(0);
                for (LongType kk = 0; kk < K; ++kk) {
                    LongType srcT = t - kk;
                    AccT xVal;
                    if (srcT >= 0) {
                        xVal = static_cast<AccT>(xBuf[b * xS0 + srcT * xS1 + d * xS2]);
                    } else if (sInBuf != nullptr) {
                        LongType stateIdx = (K - 1) + srcT;
                        xVal = stateIdx >= 0
                            ? static_cast<AccT>(sInBuf[b * siS0 + d * siS1 + stateIdx * siS2])
                            : static_cast<AccT>(0);
                    } else {
                        xVal = static_cast<AccT>(0);
                    }
                    const AccT weightValue = static_cast<AccT>(
                        wBuf[d * wChanStride + (K - 1 - kk) * wDimStride]);
                    const AccT product = reproducible::multiply<AccT>(weightValue, xVal);
                    sum = reproducible::add<AccT>(sum, product);
                }

                if (bBuf != nullptr) {
                    sum = reproducible::add<AccT>(sum, static_cast<AccT>(bBuf[d]));
                }

                if (activation == 1) {
                    const AccT one = static_cast<AccT>(1);
                    const AccT sigmoid = reproducible::divide<AccT>(
                        one, reproducible::add<AccT>(
                            one, reproducible::fastExp<AccT>(-sum)));
                    sum = reproducible::multiply<AccT>(sum, sigmoid);
                }

                outBuf[b * oS0 + t * oS1 + d * oS2] = static_cast<X>(sum);
            }

            // Update conv state from the last K-1 real timesteps, not fixed-buffer padding.
            for (LongType kk = 0; kk < K - 1; ++kk) {
                LongType srcT = effectiveLen - (K - 1) + kk;
                S val = static_cast<S>(0);
                if (srcT >= 0) {
                    val = static_cast<S>(xBuf[b * xS0 + srcT * xS1 + d * xS2]);
                } else if (sInBuf != nullptr) {
                    LongType stateIdx = (K - 1) + srcT;
                    if (stateIdx >= 0) {
                        val = sInBuf[b * siS0 + d * siS1 + stateIdx * siS2];
                    }
                }
                sOutBuf[b * soS0 + d * soS1 + kk * soS2] = val;
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, B * D);
}

void causalConv1d(LaunchContext* context, NDArray* x, NDArray* weight, NDArray* bias,
                   NDArray* stateIn, NDArray* actualLen, NDArray* output, NDArray* stateOut,
                   int activation, int wFormat) {
    NDArray::preparePrimaryUse({output, stateOut}, {x, weight, bias, actualLen});
    if (stateIn != nullptr) NDArray::preparePrimaryUse({}, {stateIn});

    const auto stateType = stateIn != nullptr ? stateIn->dataType() : x->dataType();
    BUILD_TRIPLE_SELECTOR(x->dataType(), weight->dataType(), stateType, causalConv1d_,
        (context, x, weight, bias, stateIn, actualLen, output, stateOut, activation, wFormat),
        SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output, stateOut}, {x, weight, bias, actualLen});
    if (stateIn != nullptr) NDArray::registerPrimaryUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
