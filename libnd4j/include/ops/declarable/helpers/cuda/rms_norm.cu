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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_rms_norm) || NOT_EXCLUDED(OP_skip_rms_norm) || NOT_EXCLUDED(OP_rms_norm_linear)
#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArrayFactory.h>
#include <array/DataTypeUtils.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/MmulHelper.h>
#include <ops/op_types.h>
#include <ops/declarable/helpers/rms_norm.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

constexpr int RMS_WARP_SIZE = 32;

// All normalization kernels share the registered normalization/linear launch
// family. Validate overrides before any launch; full warps are required by
// device::blockReduceSum's shuffle mask.
static dim3 rmsLaunchDims(LaunchContext* context) {
    dim3 dims = getLaunchDims("rms_norm_linear");
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, context->getDeviceID()) != cudaSuccess)
        throw std::runtime_error("RMS normalization: cannot query device launch limits");
    if (dims.x == 0 || dims.x > static_cast<unsigned int>(prop.maxGridSize[0]) ||
        dims.y == 0 || dims.y > static_cast<unsigned int>(prop.maxThreadsPerBlock) ||
        dims.y > static_cast<unsigned int>(prop.maxThreadsDim[0]) || dims.y % RMS_WARP_SIZE != 0)
        throw std::invalid_argument("RMS normalization: invalid launch dimensions (full warps required)");
    const size_t requiredShared = size_t(dims.y / RMS_WARP_SIZE) * sizeof(double);
    if (requiredShared > prop.sharedMemPerBlock || dims.z > prop.sharedMemPerBlock)
        throw std::invalid_argument("RMS normalization: insufficient shared memory");
    return dims;
}

SD_DEVICE SD_INLINE LongType rmsRowOffset(LongType row, const LongType* info) {
    const int leadingRank = shape::rank(info) - 1;
    LongType coords[SD_MAX_RANK];
    LongType offset;
    INDEX2COORDS(row, leadingRank, shape::shapeOf(info), coords);
    COORDS2INDEX(leadingRank, shape::stride(info), coords, offset);
    return offset;
}

// Optional skip/bias/hidden operands fuse the residual addition without an
// intermediate low-precision store. Each operand keeps its own view strides.
template <typename T, typename G, typename Z>
SD_KERNEL void rmsNormKernel(const T* x, const G* g, Z* z,
                            const T* skip, const T* bias, T* hidden,
                            const LongType* xInfo, const LongType* zInfo,
                            const LongType* sInfo, const LongType* hInfo,
                            LongType gs, LongType bs, LongType rows, LongType cols, double epsilon) {
    using AccT = typename simdOps::AggregateType<typename math::promote_type3<T, G, Z>::type>::type;
    extern __shared__ double rmsShared[];
    AccT* scratch = reinterpret_cast<AccT*>(rmsShared);
    __shared__ AccT inv;
    const LongType xs = shape::stride(xInfo)[shape::rank(xInfo) - 1];
    const LongType zs = shape::stride(zInfo)[shape::rank(zInfo) - 1];
    const LongType ss = skip != nullptr ? shape::stride(sInfo)[shape::rank(sInfo) - 1] : 0;
    const LongType hs = hidden != nullptr ? shape::stride(hInfo)[shape::rank(hInfo) - 1] : 0;
    for (LongType row = blockIdx.x; row < rows; row += gridDim.x) {
        const LongType xo = rmsRowOffset(row, xInfo), zo = rmsRowOffset(row, zInfo);
        const LongType so = skip != nullptr ? rmsRowOffset(row, sInfo) : 0;
        const LongType ho = hidden != nullptr ? rmsRowOffset(row, hInfo) : 0;
        AccT sumSq = 0;
        for (LongType i = threadIdx.x; i < cols; i += blockDim.x) {
            AccT v = static_cast<AccT>(x[xo + i * xs]);
            if (skip != nullptr) v += static_cast<AccT>(skip[so + i * ss]);
            if (bias != nullptr) v += static_cast<AccT>(bias[i * bs]);
            sumSq += v * v;
        }
        const AccT total = sd::device::blockReduceSum(sumSq, scratch);
        if (threadIdx.x == 0)
            inv = AccT(1) / math::sd_sqrt<AccT, AccT>(total / AccT(cols) + AccT(epsilon));
        __syncthreads();
        for (LongType i = threadIdx.x; i < cols; i += blockDim.x) {
            AccT v = static_cast<AccT>(x[xo + i * xs]);
            if (skip != nullptr) v += static_cast<AccT>(skip[so + i * ss]);
            if (bias != nullptr) v += static_cast<AccT>(bias[i * bs]);
            const AccT scale = g != nullptr ? static_cast<AccT>(g[i * gs]) : AccT(1);
            z[zo + i * zs] = static_cast<Z>(v * inv * scale);
            if (hidden != nullptr) hidden[ho + i * hs] = static_cast<T>(v);
        }
        // Shared inv/scratch must not be reused until every lane finishes.
        __syncthreads();
    }
}

template <typename T, typename G, typename Z = T>
static void launchRmsNorm(LaunchContext* context, NDArray* input, NDArray* gamma, NDArray* output,
                          NDArray* skip, NDArray* bias, NDArray* hidden, double epsilon) {
    const LongType cols = input->sizeAt(-1), rows = input->lengthOf() / cols;
    const dim3 dims = rmsLaunchDims(context);
    using AccT = typename simdOps::AggregateType<typename math::promote_type3<T, G, Z>::type>::type;
    const size_t shared = (dims.y / RMS_WARP_SIZE) * sizeof(AccT);
    const auto* xInfo = input->specialShapeInfo();
    const auto* zInfo = output->specialShapeInfo();
    const auto* sInfo = skip != nullptr ? skip->specialShapeInfo() : nullptr;
    const auto* hInfo = hidden != nullptr ? hidden->specialShapeInfo() : nullptr;
    auto* stream = context->getCudaStream();
    const unsigned int blocks = static_cast<unsigned int>(std::min<LongType>(rows, dims.x));
    rmsNormKernel<T, G, Z><<<blocks, dims.y, shared, *stream>>>(
        static_cast<const T*>(input->specialBuffer()), gamma != nullptr ? static_cast<const G*>(gamma->specialBuffer()) : nullptr,
        static_cast<Z*>(output->specialBuffer()), skip != nullptr ? static_cast<const T*>(skip->specialBuffer()) : nullptr,
        bias != nullptr ? static_cast<const T*>(bias->specialBuffer()) : nullptr,
        hidden != nullptr ? static_cast<T*>(hidden->specialBuffer()) : nullptr,
        xInfo, zInfo, sInfo, hInfo, gamma != nullptr ? gamma->strideAt(0) : 1,
        bias != nullptr ? bias->strideAt(0) : 1, rows, cols, epsilon);
    if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("rmsNormKernel failed");
}

#if NOT_EXCLUDED(OP_rms_norm)
void rmsNorm(LaunchContext* context, NDArray* input, NDArray* gamma, NDArray* output, double epsilon) {
    if (input->isEmpty()) return;
    NDArray::prepareSpecialUse({output}, {input, gamma});
    const auto gammaType = gamma != nullptr ? gamma->dataType() : input->dataType();
    BUILD_DOUBLE_SELECTOR(input->dataType(), gammaType, launchRmsNorm,
                          (context, input, gamma, output, nullptr, nullptr, nullptr, epsilon),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input, gamma});
}
#endif

#if NOT_EXCLUDED(OP_skip_rms_norm)
void skipRmsNorm(LaunchContext* context, NDArray* input, NDArray* skip, NDArray* gamma,
                 NDArray* bias, NDArray* output, NDArray* hiddenOut, double epsilon) {
    if (input->isEmpty()) return;
    NDArray::prepareSpecialUse({output, hiddenOut}, {input, skip, gamma, bias});
    BUILD_DOUBLE_SELECTOR(input->dataType(), gamma->dataType(), launchRmsNorm,
                          (context, input, gamma, output, skip, bias, hiddenOut, epsilon),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output, hiddenOut}, {input, skip, gamma, bias});
}
#endif

#if NOT_EXCLUDED(OP_rms_norm_linear)
// Decode: normalized activations remain in AccT shared memory. Weights and
// gamma are read in their native dtypes, with no cast cache or weight copy.
template <typename T, typename G, typename W>
SD_KERNEL void rmsNormLinearFusedKernel(const T* x, const G* gamma, const W* weight, T* output,
                                       LongType K, LongType N, LongType xs, LongType gs, LongType zs,
                                       LongType ws0, LongType ws1, double epsilon) {
    using AccT = typename simdOps::AggregateType<typename math::promote_type3<T, G, W>::type>::type;
    extern __shared__ double linearShared[];
    AccT* normX = reinterpret_cast<AccT*>(linearShared);
    AccT* scratch = normX + K;
    AccT sumSq = 0;
    for (LongType i = threadIdx.x; i < K; i += blockDim.x) {
        const AccT v = static_cast<AccT>(x[i * xs]);
        sumSq += v * v;
    }
    const AccT total = sd::device::blockReduceSum(sumSq, scratch);
    __shared__ AccT inv;
    if (threadIdx.x == 0)
        inv = AccT(1) / math::sd_sqrt<AccT, AccT>(total / AccT(K) + AccT(epsilon));
    __syncthreads();
    for (LongType i = threadIdx.x; i < K; i += blockDim.x) {
        const AccT scale = gamma != nullptr ? static_cast<AccT>(gamma[i * gs]) : AccT(1);
        normX[i] = static_cast<AccT>(x[i * xs]) * inv * scale;
    }
    __syncthreads();
    for (LongType j = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < N; j += static_cast<LongType>(gridDim.x) * blockDim.x) {
        AccT acc = 0;
        for (LongType k = 0; k < K; ++k)
            acc += normX[k] * static_cast<AccT>(weight[k * ws0 + j * ws1]);
        output[j * zs] = static_cast<T>(acc);
    }
}

template <typename T, typename G, typename W>
static void rmsNormLinearLauncher(LaunchContext* context, NDArray* input, NDArray* gamma,
                                  NDArray* weight, NDArray* output, double epsilon) {
    using AccT = typename simdOps::AggregateType<typename math::promote_type3<T, G, W>::type>::type;
    const LongType M = input->sizeAt(0), K = input->sizeAt(1), N = weight->sizeAt(1);
    dim3 dims = rmsLaunchDims(context);
    const size_t requiredShared = std::max<size_t>(dims.z, (K + dims.y / RMS_WARP_SIZE) * sizeof(AccT));
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, context->getDeviceID()) != cudaSuccess)
        throw std::runtime_error("RMS normalization: cannot query shared memory limit");
    // Admission, not post-launch recovery: larger rows use the existing
    // normalization + BLAS algorithm without exceeding device shared memory.
    if (M == 1 && K <= 8192 && requiredShared <= prop.sharedMemPerBlock) {
        NDArray::prepareSpecialUse({output}, {input, gamma, weight});
        auto* stream = context->getCudaStream();
        const unsigned int blocks = static_cast<unsigned int>(std::min<LongType>(dims.x, (N - 1) / dims.y + 1));
        rmsNormLinearFusedKernel<T, G, W><<<blocks, dims.y, requiredShared, *stream>>>(
            static_cast<const T*>(input->specialBuffer()), gamma != nullptr ? static_cast<const G*>(gamma->specialBuffer()) : nullptr,
            static_cast<const W*>(weight->specialBuffer()), static_cast<T*>(output->specialBuffer()),
            K, N, input->strideAt(-1), gamma != nullptr ? gamma->strideAt(0) : 1,
            output->strideAt(-1), weight->strideAt(0), weight->strideAt(1), epsilon);
        if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("rmsNormLinearFusedKernel failed");
        NDArray::registerSpecialUse({output}, {input, gamma, weight});
        return;
    }

    // Prefill keeps normalization and GEMM intermediates in AccT, narrowing
    // only at the final output store, just like the fused decode kernel.
    // BLAS requires matching operand dtypes; bound weight conversion to
    // reusable panels rather than materializing a full widened model matrix.
    const auto calcType = DataTypeUtils::fromT<AccT>();
    std::vector<LongType> normalizedShape = {M, K};
    NDArray normalized('c', normalizedShape, calcType, context);
    NDArray::prepareSpecialUse({&normalized}, {input, gamma});
    launchRmsNorm<T, G, AccT>(context, input, gamma, &normalized, nullptr, nullptr, nullptr, epsilon);
    NDArray::registerSpecialUse({&normalized}, {input, gamma});
    if (weight->dataType() == calcType && output->dataType() == calcType) {
        MmulHelper::mmul(&normalized, weight, output, 1.0, 0.0);
        return;
    }
    constexpr LongType panelBytes = 16LL * 1024 * 1024;
    const LongType columns = std::min(N, std::max<LongType>(1, panelBytes / sizeof(AccT) / (K + M)));
    std::vector<LongType> weightShape = {K, columns}, resultShape = {M, columns};
    NDArray weightPanel('f', weightShape, calcType, context);
    NDArray resultPanel('f', resultShape, calcType, context);
    for (LongType col = 0; col < N; col += columns) {
        const LongType count = std::min(columns, N - col);
        // ResultSet deliberately does not delete views. These wrappers belong
        // to this panel iteration, while their buffers belong to the parents.
        NDArray *source = nullptr, *panel = nullptr, *result = nullptr, *destination = nullptr;
        auto releaseViews = [&]() {
            delete destination;
            delete result;
            delete panel;
            delete source;
        };
        try {
            source = (*weight)({0, K, col, col + count}, true);
            panel = weightPanel({0, K, 0, count}, true);
            result = resultPanel({0, M, 0, count}, true);
            destination = (*output)({0, M, col, col + count}, true);
            panel->assign(source);
            MmulHelper::mmul(&normalized, panel, result, 1.0, 0.0);
            destination->assign(result);
        } catch (...) {
            releaseViews();
            throw;
        }
        releaseViews();
    }
}

void rmsNormLinear(LaunchContext* context, NDArray* input, NDArray* gamma, NDArray* weight,
                   NDArray* output, double epsilon) {
    if (output->isEmpty()) return;
    if (input->isEmpty()) THROW_EXCEPTION("rmsNormLinear: nonempty output requires nonempty input");
    const auto gammaType = gamma != nullptr ? gamma->dataType() : input->dataType();
    BUILD_TRIPLE_SELECTOR(input->dataType(), gammaType, weight->dataType(), rmsNormLinearLauncher,
                          (context, input, gamma, weight, output, epsilon),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_FLOAT_TYPES);
}
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
