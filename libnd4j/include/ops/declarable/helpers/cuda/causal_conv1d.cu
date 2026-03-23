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
#include <types/float16.h>
#include <ops/declarable/helpers/causal_conv1d.h>

namespace sd {
namespace ops {
namespace helpers {

// One thread per (batch, time, channel) element
template <typename T>
SD_KERNEL void causalConv1dKernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ stateIn,
    T* __restrict__ out,
    const LongType B, const LongType L, const LongType D, const LongType K,
    const int activation,
    const LongType xS0, const LongType xS1, const LongType xS2,
    const LongType wS0, const LongType wS1,
    const LongType oS0, const LongType oS1, const LongType oS2,
    const LongType siS0, const LongType siS1, const LongType siS2) {

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const LongType total = B * L * D;
    if (idx >= total) return;

    const LongType d = idx % D;
    const LongType t = (idx / D) % L;
    const LongType b = idx / (L * D);

    T sum = static_cast<T>(0);
    for (LongType kk = 0; kk < K; ++kk) {
        LongType srcT = t - kk;
        T x_val;
        if (srcT >= 0) {
            x_val = x[b * xS0 + srcT * xS1 + d * xS2];
        } else if (stateIn != nullptr) {
            LongType stateIdx = (K - 1) + srcT;
            x_val = (stateIdx >= 0) ? stateIn[b * siS0 + d * siS1 + stateIdx * siS2] : static_cast<T>(0);
        } else {
            x_val = static_cast<T>(0);
        }
        sum += weight[d * wS0 + kk * wS1] * x_val;
    }

    if (bias != nullptr) sum += bias[d];

    // SiLU activation
    if (activation == 1) {
        T sig = static_cast<T>(1) / (static_cast<T>(1) + sd::math::sd_exp<T, T>(-sum));
        sum = sum * sig;
    }

    out[b * oS0 + t * oS1 + d * oS2] = sum;
}

// Update conv state: store last K-1 timesteps per channel
template <typename T>
SD_KERNEL void convStateUpdateKernel(
    const T* __restrict__ x,
    const T* __restrict__ stateIn,
    T* __restrict__ stateOut,
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

    LongType srcT = L - (K - 1) + kk;
    T val = static_cast<T>(0);
    if (srcT >= 0) {
        val = x[b * xS0 + srcT * xS1 + d * xS2];
    } else if (stateIn != nullptr) {
        LongType stateIdx = (K - 1) + srcT;
        if (stateIdx >= 0) val = stateIn[b * siS0 + d * siS1 + stateIdx * siS2];
    }
    stateOut[b * soS0 + d * soS1 + kk * soS2] = val;
}

template <typename T>
static void launchCausalConv1d(
    const T* x, const T* weight, const T* bias, const T* stateIn,
    T* out, T* stateOut,
    LongType B, LongType L, LongType D, LongType K, int activation,
    LongType xS0, LongType xS1, LongType xS2,
    LongType wS0, LongType wS1,
    LongType oS0, LongType oS1, LongType oS2,
    LongType siS0, LongType siS1, LongType siS2,
    LongType soS0, LongType soS1, LongType soS2,
    cudaStream_t stream) {

    int threadsPerBlock = 256;

    LongType total = B * L * D;
    int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    causalConv1dKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        x, weight, bias, stateIn, out, B, L, D, K, activation,
        xS0, xS1, xS2, wS0, wS1, oS0, oS1, oS2, siS0, siS1, siS2);
    DebugHelper::checkGlobalErrorCode("causalConv1dKernel failed");

    LongType stateTotal = B * D * (K - 1);
    int stateBlocks = (stateTotal + threadsPerBlock - 1) / threadsPerBlock;
    convStateUpdateKernel<T><<<stateBlocks, threadsPerBlock, 0, stream>>>(
        x, stateIn, stateOut, B, L, D, K,
        xS0, xS1, xS2, siS0, siS1, siS2, soS0, soS1, soS2);
    DebugHelper::checkGlobalErrorCode("convStateUpdateKernel failed");
}

// No explicit instantiation needed — launchCausalConv1d is file-local and called via type switch below.

void causalConv1d(LaunchContext* context, NDArray* x, NDArray* weight, NDArray* bias,
                   NDArray* stateIn, NDArray* output, NDArray* stateOut, int activation) {
    const auto B = x->sizeAt(0);
    const auto L = x->sizeAt(1);
    const auto D = x->sizeAt(2);
    const auto K = weight->sizeAt(1);

    NDArray::prepareSpecialUse({output, stateOut}, {x, weight, bias});
    if (stateIn != nullptr) NDArray::prepareSpecialUse({}, {stateIn});

    auto stream = context->getCudaStream();
    auto dtype = x->dataType();

    LongType siS0 = 0, siS1 = 0, siS2 = 0;
    if (stateIn != nullptr) {
        siS0 = stateIn->strideAt(0);
        siS1 = stateIn->strideAt(1);
        siS2 = stateIn->strideAt(2);
    }

    if (dtype == DataType::FLOAT32) {
        launchCausalConv1d<float>(
            reinterpret_cast<const float*>(x->specialBuffer()),
            reinterpret_cast<const float*>(weight->specialBuffer()),
            bias ? reinterpret_cast<const float*>(bias->specialBuffer()) : nullptr,
            stateIn ? reinterpret_cast<const float*>(stateIn->specialBuffer()) : nullptr,
            reinterpret_cast<float*>(output->specialBuffer()),
            reinterpret_cast<float*>(stateOut->specialBuffer()),
            B, L, D, K, activation,
            x->strideAt(0), x->strideAt(1), x->strideAt(2),
            weight->strideAt(0), weight->strideAt(1),
            output->strideAt(0), output->strideAt(1), output->strideAt(2),
            siS0, siS1, siS2,
            stateOut->strideAt(0), stateOut->strideAt(1), stateOut->strideAt(2),
            *stream);
    } else if (dtype == DataType::DOUBLE) {
        launchCausalConv1d<double>(
            reinterpret_cast<const double*>(x->specialBuffer()),
            reinterpret_cast<const double*>(weight->specialBuffer()),
            bias ? reinterpret_cast<const double*>(bias->specialBuffer()) : nullptr,
            stateIn ? reinterpret_cast<const double*>(stateIn->specialBuffer()) : nullptr,
            reinterpret_cast<double*>(output->specialBuffer()),
            reinterpret_cast<double*>(stateOut->specialBuffer()),
            B, L, D, K, activation,
            x->strideAt(0), x->strideAt(1), x->strideAt(2),
            weight->strideAt(0), weight->strideAt(1),
            output->strideAt(0), output->strideAt(1), output->strideAt(2),
            siS0, siS1, siS2,
            stateOut->strideAt(0), stateOut->strideAt(1), stateOut->strideAt(2),
            *stream);
    } else if (dtype == DataType::HALF) {
        launchCausalConv1d<float16>(
            reinterpret_cast<const float16*>(x->specialBuffer()),
            reinterpret_cast<const float16*>(weight->specialBuffer()),
            bias ? reinterpret_cast<const float16*>(bias->specialBuffer()) : nullptr,
            stateIn ? reinterpret_cast<const float16*>(stateIn->specialBuffer()) : nullptr,
            reinterpret_cast<float16*>(output->specialBuffer()),
            reinterpret_cast<float16*>(stateOut->specialBuffer()),
            B, L, D, K, activation,
            x->strideAt(0), x->strideAt(1), x->strideAt(2),
            weight->strideAt(0), weight->strideAt(1),
            output->strideAt(0), output->strideAt(1), output->strideAt(2),
            siS0, siS1, siS2,
            stateOut->strideAt(0), stateOut->strideAt(1), stateOut->strideAt(2),
            *stream);
    } else {
        THROW_EXCEPTION("causalConv1d: Unsupported data type");
    }

    NDArray::registerSpecialUse({output, stateOut}, {x, weight, bias});
    if (stateIn != nullptr) NDArray::registerSpecialUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
