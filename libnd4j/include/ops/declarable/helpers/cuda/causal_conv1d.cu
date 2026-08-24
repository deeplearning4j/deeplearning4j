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

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <math/templatemath.h>
#include <ops/op_types.h>
#include <types/float16.h>
#include <ops/declarable/helpers/causal_conv1d.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>
#include <ops/declarable/helpers/reproducible_math.h>

namespace sd {
namespace ops {
namespace helpers {

// One thread per (batch, time, channel) element. Activations/state/output use X;
// weights may use a lower-precision storage type W.
template <typename X, typename W, typename S>
SD_KERNEL void causalConv1dKernel(
    const X* __restrict__ x,
    const W* __restrict__ weight,
    const X* __restrict__ bias,
    const S* __restrict__ stateIn,
    X* __restrict__ out,
    const LongType B, const LongType L, const LongType D, const LongType K,
    const int activation,
    const LongType xS0, const LongType xS1, const LongType xS2,
    const LongType wChanStride, const LongType wDimStride,
    const LongType oS0, const LongType oS1, const LongType oS2,
    const LongType siS0, const LongType siS1, const LongType siS2) {

    using PromotedT = typename sd::math::promote_type3<X, W, S>::type;
    using AccT = typename simdOps::AggregateType<PromotedT>::type;

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const LongType total = B * L * D;
    if (idx >= total) return;

    const LongType d = idx % D;
    const LongType t = (idx / D) % L;
    const LongType b = idx / (L * D);

    // Causal convolution matching PyTorch F.conv1d with left-padding:
    //   F.conv1d(x, w.unsqueeze(1), padding=K-1)[:, :, :L]
    // weight[K-1] multiplies x[t] (current), weight[0] multiplies x[t-K+1] (oldest)
    // Accumulate in AccT (double for T=double, float otherwise) for precision.
    AccT sum = static_cast<AccT>(0);
    for (LongType kk = 0; kk < K; ++kk) {
        LongType srcT = t - kk;
        AccT x_val;
        if (srcT >= 0) {
            x_val = static_cast<AccT>(x[b * xS0 + srcT * xS1 + d * xS2]);
        } else if (stateIn != nullptr) {
            LongType stateIdx = (K - 1) + srcT;
            x_val = (stateIdx >= 0) ? static_cast<AccT>(stateIn[b * siS0 + d * siS1 + stateIdx * siS2]) : static_cast<AccT>(0);
        } else {
            x_val = static_cast<AccT>(0);
        }
        const AccT weightValue = static_cast<AccT>(
            weight[d * wChanStride + (K - 1 - kk) * wDimStride]);
        const AccT product = reproducible::multiply<AccT>(weightValue, x_val);
        sum = reproducible::add<AccT>(sum, product);
    }

    if (bias != nullptr) {
        sum = reproducible::add<AccT>(sum, static_cast<AccT>(bias[d]));
    }

    if (activation == 1) {
        const AccT one = static_cast<AccT>(1);
        const AccT sigmoid = reproducible::divide<AccT>(
            one, reproducible::add<AccT>(
                one, reproducible::fastExp<AccT>(-sum)));
        sum = reproducible::multiply<AccT>(sum, sigmoid);
    }

    out[b * oS0 + t * oS1 + d * oS2] = static_cast<X>(sum);
}

// Update conv state: store last K-1 timesteps per channel
template <typename X, typename S>
SD_KERNEL void convStateUpdateKernel(
    const X* __restrict__ x,
    const S* __restrict__ stateIn,
    const LongType* __restrict__ actualLen,
    S* __restrict__ stateOut,
    const LongType B, const LongType L, const LongType D, const LongType K,
    const LongType xS0, const LongType xS1, const LongType xS2,
    const LongType siS0, const LongType siS1, const LongType siS2,
    const LongType soS0, const LongType soS1, const LongType soS2) {

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const LongType total = B * D * (K - 1);
    if (idx >= total) return;

    const LongType kk = idx % (K - 1);
    const LongType d = (idx / (K - 1)) % D;
    const LongType b = idx / (D * (K - 1));

    LongType effectiveLen = L;
    if (actualLen != nullptr) {
        effectiveLen = actualLen[0];
        if (effectiveLen < 0) effectiveLen = 0;
        if (effectiveLen > L) effectiveLen = L;
    }
    LongType srcT = effectiveLen - (K - 1) + kk;
    S val = static_cast<S>(0);
    if (srcT >= 0) {
        val = static_cast<S>(x[b * xS0 + srcT * xS1 + d * xS2]);
    } else if (stateIn != nullptr) {
        LongType stateIdx = (K - 1) + srcT;
        if (stateIdx >= 0) {
            val = stateIn[b * siS0 + d * siS1 + stateIdx * siS2];
        }
    }
    stateOut[b * soS0 + d * soS1 + kk * soS2] = val;
}

template <typename X, typename W, typename S>
static void launchCausalConv1d(
    const X* x, const W* weight, const X* bias, const S* stateIn,
    const LongType* actualLen,
    X* out, S* stateOut,
    LongType B, LongType L, LongType D, LongType K, int activation,
    LongType xS0, LongType xS1, LongType xS2,
    LongType wChanStride, LongType wDimStride,
    LongType oS0, LongType oS1, LongType oS2,
    LongType siS0, LongType siS1, LongType siS2,
    LongType soS0, LongType soS1, LongType soS2,
    cudaStream_t stream) {

    int threadsPerBlock = 256;

    LongType total = B * L * D;
    int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    causalConv1dKernel<X, W, S><<<numBlocks, threadsPerBlock, 0, stream>>>(
        x, weight, bias, stateIn, out, B, L, D, K, activation,
        xS0, xS1, xS2, wChanStride, wDimStride, oS0, oS1, oS2, siS0, siS1, siS2);
    DebugHelper::checkGlobalErrorCode("causalConv1dKernel failed");

    LongType stateTotal = B * D * (K - 1);
    if (stateTotal > 0) {
        int stateBlocks = (stateTotal + threadsPerBlock - 1) / threadsPerBlock;
        convStateUpdateKernel<X, S><<<stateBlocks, threadsPerBlock, 0, stream>>>(
            x, stateIn, actualLen, stateOut, B, L, D, K,
            xS0, xS1, xS2, siS0, siS1, siS2, soS0, soS1, soS2);
        DebugHelper::checkGlobalErrorCode("convStateUpdateKernel failed");
    }
}

// No explicit instantiation needed — the selector below instantiates this file-local launcher.

template <typename X, typename W, typename S>
static void launchCausalConv1dFromArrays(
    LaunchContext* context, NDArray* x, NDArray* weight, NDArray* bias,
    NDArray* stateIn, NDArray* actualLen, NDArray* output, NDArray* stateOut,
    int activation, int wFormat) {
    const auto B = x->sizeAt(0);
    const auto L = x->sizeAt(1);
    const auto D = x->sizeAt(2);
    const auto K = (wFormat == 0) ? weight->sizeAt(1) : weight->sizeAt(0);
    const LongType wChanStride = (wFormat == 0) ? weight->strideAt(0) : weight->strideAt(1);
    const LongType wDimStride = (wFormat == 0) ? weight->strideAt(1) : weight->strideAt(0);

    LongType siS0 = 0, siS1 = 0, siS2 = 0;
    if (stateIn != nullptr) {
        siS0 = stateIn->strideAt(0);
        siS1 = stateIn->strideAt(1);
        siS2 = stateIn->strideAt(2);
    }

    launchCausalConv1d<X, W, S>(
        reinterpret_cast<const X*>(x->specialBuffer()),
        reinterpret_cast<const W*>(weight->specialBuffer()),
        bias ? reinterpret_cast<const X*>(bias->specialBuffer()) : nullptr,
        stateIn ? reinterpret_cast<const S*>(stateIn->specialBuffer()) : nullptr,
        actualLen ? reinterpret_cast<const LongType*>(actualLen->specialBuffer()) : nullptr,
        reinterpret_cast<X*>(output->specialBuffer()),
        reinterpret_cast<S*>(stateOut->specialBuffer()),
        B, L, D, K, activation,
        x->strideAt(0), x->strideAt(1), x->strideAt(2),
        wChanStride, wDimStride,
        output->strideAt(0), output->strideAt(1), output->strideAt(2),
        siS0, siS1, siS2,
        stateOut->strideAt(0), stateOut->strideAt(1), stateOut->strideAt(2),
        *context->getCudaStream());
}

void causalConv1d(LaunchContext* context, NDArray* x, NDArray* weight, NDArray* bias,
                   NDArray* stateIn, NDArray* actualLen, NDArray* output, NDArray* stateOut,
                   int activation, int wFormat) {
    NDArray::prepareSpecialUse({output, stateOut}, {x, weight, bias, actualLen});
    if (stateIn != nullptr) NDArray::prepareSpecialUse({}, {stateIn});

    const auto stateType = stateIn != nullptr ? stateIn->dataType() : x->dataType();
    BUILD_TRIPLE_SELECTOR(
        x->dataType(), weight->dataType(), stateType, launchCausalConv1dFromArrays,
        (context, x, weight, bias, stateIn, actualLen, output, stateOut, activation, wFormat),
        SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output, stateOut}, {x, weight, bias, actualLen});
    if (stateIn != nullptr) NDArray::registerSpecialUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
