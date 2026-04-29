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
#include <ops/declarable/helpers/gated_delta_rule.h>

namespace sd {
namespace ops {
namespace helpers {

// One block per (batch, head). Threads cover D_v dimension.
// Sequential over timesteps (recurrent dependency).
template <typename T>
SD_KERNEL void gatedDeltaRuleKernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const T* __restrict__ betaArr,
    const T* __restrict__ gateArr,
    T* __restrict__ state,
    T* __restrict__ out,
    const LongType B, const LongType L, const LongType H,
    const LongType D_k, const LongType D_v,
    const LongType t,
    const LongType qS0, const LongType qS1, const LongType qS2, const LongType qS3,
    const LongType kS0, const LongType kS1, const LongType kS2, const LongType kS3,
    const LongType vS0, const LongType vS1, const LongType vS2, const LongType vS3,
    const LongType bS0, const LongType bS1, const LongType bS2,
    const LongType gS0, const LongType gS1, const LongType gS2,
    const LongType oS0, const LongType oS1, const LongType oS2, const LongType oS3) {

    const LongType bh = blockIdx.x;
    if (bh >= B * H) return;

    const LongType b = bh / H;
    const LongType h = bh % H;

    const T exp_g = sd::math::sd_exp<T, T>(gateArr[b * gS0 + t * gS1 + h * gS2]);
    const T beta_val = betaArr[b * bS0 + t * bS1 + h * bS2];
    T* sPtr = state + (b * H + h) * D_k * D_v;

    // Use float accumulators for dot products to prevent HALF overflow.
    // D_k=128 terms accumulated in float16 can exceed 65504 (FP16 max).
    const float exp_g_f = static_cast<float>(exp_g);
    const float beta_f = static_cast<float>(beta_val);

    for (LongType dv = threadIdx.x; dv < D_v; dv += blockDim.x) {
        // prediction = S^T * k  (accumulated in float32)
        float prediction = 0.0f;
        for (LongType dk = 0; dk < D_k; ++dk)
            prediction += static_cast<float>(sPtr[dk * D_v + dv]) * static_cast<float>(k[b * kS0 + t * kS1 + h * kS2 + dk * kS3]);

        // delta = v - exp(g) * prediction
        const float delta = static_cast<float>(v[b * vS0 + t * vS1 + h * vS2 + dv * vS3]) - exp_g_f * prediction;

        // S = exp(g) * S + beta * k * delta
        for (LongType dk = 0; dk < D_k; ++dk) {
            const float k_val = static_cast<float>(k[b * kS0 + t * kS1 + h * kS2 + dk * kS3]);
            sPtr[dk * D_v + dv] = static_cast<T>(exp_g_f * static_cast<float>(sPtr[dk * D_v + dv]) + beta_f * k_val * delta);
        }

        // output = S^T * q  (accumulated in float32)
        float out_val = 0.0f;
        for (LongType dk = 0; dk < D_k; ++dk)
            out_val += static_cast<float>(sPtr[dk * D_v + dv]) * static_cast<float>(q[b * qS0 + t * qS1 + h * qS2 + dk * qS3]);
        out[b * oS0 + t * oS1 + h * oS2 + dv * oS3] = static_cast<T>(out_val);
    }
}

template <typename T>
static void launchGatedDeltaRule(
    const T* q, const T* k, const T* v,
    const T* betaArr, const T* gateArr,
    T* state, T* out,
    LongType B, LongType L, LongType H, LongType D_k, LongType D_v,
    LongType qS0, LongType qS1, LongType qS2, LongType qS3,
    LongType kS0, LongType kS1, LongType kS2, LongType kS3,
    LongType vS0, LongType vS1, LongType vS2, LongType vS3,
    LongType bS0, LongType bS1, LongType bS2,
    LongType gS0, LongType gS1, LongType gS2,
    LongType oS0, LongType oS1, LongType oS2, LongType oS3,
    cudaStream_t stream) {

    int numBlocks = B * H;
    int threadsPerBlock = 256;
    if (D_v < threadsPerBlock) {
        threadsPerBlock = ((D_v + 31) / 32) * 32;
        if (threadsPerBlock < 32) threadsPerBlock = 32;
    }

    for (LongType t = 0; t < L; ++t) {
        gatedDeltaRuleKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
            q, k, v, betaArr, gateArr, state, out,
            B, L, H, D_k, D_v, t,
            qS0, qS1, qS2, qS3, kS0, kS1, kS2, kS3,
            vS0, vS1, vS2, vS3, bS0, bS1, bS2,
            gS0, gS1, gS2, oS0, oS1, oS2, oS3);
    }
    DebugHelper::checkGlobalErrorCode("gatedDeltaRuleKernel failed");
}

// No explicit instantiation needed — launchGatedDeltaRule is file-local and called via BUILD_SINGLE_SELECTOR below.

void gatedDeltaRule(LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                     NDArray* beta, NDArray* gate, NDArray* stateIn,
                     NDArray* output, NDArray* stateOut) {
    const auto B = Q->sizeAt(0);
    const auto L = Q->sizeAt(1);
    const auto H = Q->sizeAt(2);
    const auto D_k = Q->sizeAt(3);
    const auto D_v = V->sizeAt(3);

    NDArray::prepareSpecialUse({output, stateOut}, {Q, K, V, beta, gate});
    if (stateIn != nullptr) NDArray::prepareSpecialUse({}, {stateIn});

    auto stream = context->getCudaStream();

    stateOut->nullify();
    if (stateIn != nullptr) stateOut->assign(stateIn);

    auto dtype = Q->dataType();
    if (dtype == DataType::FLOAT32) {
        launchGatedDeltaRule<float>(
            reinterpret_cast<const float*>(Q->specialBuffer()),
            reinterpret_cast<const float*>(K->specialBuffer()),
            reinterpret_cast<const float*>(V->specialBuffer()),
            reinterpret_cast<const float*>(beta->specialBuffer()),
            reinterpret_cast<const float*>(gate->specialBuffer()),
            reinterpret_cast<float*>(stateOut->specialBuffer()),
            reinterpret_cast<float*>(output->specialBuffer()),
            B, L, H, D_k, D_v,
            Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
            K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
            V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
            beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
            gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
            output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
            *stream);
    } else if (dtype == DataType::DOUBLE) {
        launchGatedDeltaRule<double>(
            reinterpret_cast<const double*>(Q->specialBuffer()),
            reinterpret_cast<const double*>(K->specialBuffer()),
            reinterpret_cast<const double*>(V->specialBuffer()),
            reinterpret_cast<const double*>(beta->specialBuffer()),
            reinterpret_cast<const double*>(gate->specialBuffer()),
            reinterpret_cast<double*>(stateOut->specialBuffer()),
            reinterpret_cast<double*>(output->specialBuffer()),
            B, L, H, D_k, D_v,
            Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
            K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
            V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
            beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
            gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
            output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
            *stream);
    } else if (dtype == DataType::HALF) {
        launchGatedDeltaRule<float16>(
            reinterpret_cast<const float16*>(Q->specialBuffer()),
            reinterpret_cast<const float16*>(K->specialBuffer()),
            reinterpret_cast<const float16*>(V->specialBuffer()),
            reinterpret_cast<const float16*>(beta->specialBuffer()),
            reinterpret_cast<const float16*>(gate->specialBuffer()),
            reinterpret_cast<float16*>(stateOut->specialBuffer()),
            reinterpret_cast<float16*>(output->specialBuffer()),
            B, L, H, D_k, D_v,
            Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
            K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
            V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
            beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
            gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
            output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
            *stream);
    } else {
        THROW_EXCEPTION("gatedDeltaRule: Unsupported data type");
    }

    NDArray::registerSpecialUse({output, stateOut}, {Q, K, V, beta, gate});
    if (stateIn != nullptr) NDArray::registerSpecialUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
