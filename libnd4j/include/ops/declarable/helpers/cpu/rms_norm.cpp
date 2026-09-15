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
#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/op_types.h>
#include <array/NDArrayFactory.h>
#include <array/DataTypeUtils.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/rms_norm.h>

namespace sd {
namespace ops {
namespace helpers {

// Rows are logical C-order leading coordinates, not row * strideAt(-2).
// Buffer pointers already include the view base offset.
static LongType rmsRowOffset(LongType row, const LongType* info) {
    const int leadingRank = shape::rank(info) - 1;
    LongType coords[SD_MAX_RANK];
    LongType offset;
    INDEX2COORDS(row, leadingRank, shape::shapeOf(info), coords);
    COORDS2INDEX(leadingRank, shape::stride(info), coords, offset);
    return offset;
}

template <typename T, typename G, typename Z = T>
static void rmsNorm_(NDArray* input, NDArray* gamma, NDArray* output, double epsilon) {
    if (input->isEmpty()) return;
    const LongType rowLen = input->sizeAt(-1);
    const LongType numRows = input->lengthOf() / rowLen;
    const LongType xs = input->strideAt(-1), zs = output->strideAt(-1);
    const LongType gs = gamma != nullptr ? gamma->strideAt(0) : 1;
    const T* x = input->bufferAsT<T>();
    const G* g = gamma != nullptr ? gamma->bufferAsT<G>() : nullptr;
    Z* z = output->bufferAsT<Z>();
    using AccT = typename simdOps::AggregateType<typename math::promote_type3<T, G, Z>::type>::type;
    auto func = PRAGMA_THREADS_FOR {
        for (LongType row = start; row < stop; row += increment) {
            const LongType xo = rmsRowOffset(row, input->shapeInfo());
            const LongType zo = rmsRowOffset(row, output->shapeInfo());
            AccT sumSq = 0;
            for (LongType i = 0; i < rowLen; ++i) {
                const AccT v = static_cast<AccT>(x[xo + i * xs]);
                sumSq += v * v;
            }
            const AccT inv = AccT(1) / math::sd_sqrt<AccT, AccT>(sumSq / AccT(rowLen) + AccT(epsilon));
            for (LongType i = 0; i < rowLen; ++i) {
                const AccT scale = g != nullptr ? static_cast<AccT>(g[i * gs]) : AccT(1);
                z[zo + i * zs] = static_cast<Z>(static_cast<AccT>(x[xo + i * xs]) * inv * scale);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
}

#if NOT_EXCLUDED(OP_rms_norm)
void rmsNorm(LaunchContext* context, NDArray* input, NDArray* gamma, NDArray* output, double epsilon) {
    if (input->isEmpty()) return;
    NDArray::preparePrimaryUse({output}, {input, gamma});
    const auto gammaType = gamma != nullptr ? gamma->dataType() : input->dataType();
    BUILD_DOUBLE_SELECTOR(input->dataType(), gammaType, rmsNorm_, (input, gamma, output, epsilon),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES);
    NDArray::registerPrimaryUse({output}, {input, gamma});
}
#endif

#if NOT_EXCLUDED(OP_rms_norm_linear)
template <typename T, typename G, typename W>
static void rmsNormLinear_(LaunchContext* context, NDArray* input, NDArray* gamma, NDArray* weight,
                           NDArray* output, double epsilon) {
    using AccT = typename simdOps::AggregateType<typename math::promote_type3<T, G, W>::type>::type;
    const auto calcType = DataTypeUtils::fromT<AccT>();
    const LongType M = input->sizeAt(0), K = input->sizeAt(1), N = weight->sizeAt(1);
    std::vector<LongType> normalizedShape = {M, K};
    NDArray normalized('c', normalizedShape, calcType, context);
    NDArray::preparePrimaryUse({&normalized}, {input, gamma});
    rmsNorm_<T, G, AccT>(input, gamma, &normalized, epsilon);
    NDArray::registerPrimaryUse({&normalized}, {input, gamma});
    if (weight->dataType() == calcType && output->dataType() == calcType) {
        MmulHelper::mmul(&normalized, weight, output, 1.0, 0.0);
        return;
    }

    // BLAS consumes matching dtypes. Bound conversion workspace to a panel;
    // never materialize a full widened model weight matrix.
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

void rmsNormLinear(LaunchContext* context, NDArray* input, NDArray* gamma,
                   NDArray* weight, NDArray* output, double epsilon) {
    if (output->isEmpty()) return;
    if (input->isEmpty()) THROW_EXCEPTION("rmsNormLinear: nonempty output requires nonempty input");
    const auto gammaType = gamma != nullptr ? gamma->dataType() : input->dataType();
    BUILD_TRIPLE_SELECTOR(input->dataType(), gammaType, weight->dataType(), rmsNormLinear_,
                          (context, input, gamma, weight, output, epsilon),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_FLOAT_TYPES);
}
#endif

#if NOT_EXCLUDED(OP_skip_rms_norm)
template <typename T, typename G>
static void skipRmsNorm_(NDArray* input, NDArray* skip, NDArray* gamma, NDArray* bias,
                         NDArray* output, NDArray* hiddenOut, double epsilon) {
    const LongType rowLen = input->sizeAt(-1);
    const LongType numRows = input->lengthOf() / rowLen;
    const LongType xs = input->strideAt(-1), ss = skip->strideAt(-1), zs = output->strideAt(-1);
    const LongType hs = hiddenOut != nullptr ? hiddenOut->strideAt(-1) : 1;
    const LongType gs = gamma->strideAt(0), bs = bias != nullptr ? bias->strideAt(0) : 1;
    const T* x = input->bufferAsT<T>();
    const T* s = skip->bufferAsT<T>();
    const T* b = bias != nullptr ? bias->bufferAsT<T>() : nullptr;
    const G* g = gamma->bufferAsT<G>();
    T* z = output->bufferAsT<T>();
    T* h = hiddenOut != nullptr ? hiddenOut->bufferAsT<T>() : nullptr;
    using AccT = typename simdOps::AggregateType<typename math::promote_type<T, G>::type>::type;
    auto func = PRAGMA_THREADS_FOR {
        for (LongType row = start; row < stop; row += increment) {
            const LongType xo = rmsRowOffset(row, input->shapeInfo());
            const LongType so = rmsRowOffset(row, skip->shapeInfo());
            const LongType zo = rmsRowOffset(row, output->shapeInfo());
            const LongType ho = h != nullptr ? rmsRowOffset(row, hiddenOut->shapeInfo()) : 0;
            AccT sumSq = 0;
            for (LongType i = 0; i < rowLen; ++i) {
                AccT v = static_cast<AccT>(x[xo + i * xs]) + static_cast<AccT>(s[so + i * ss]);
                if (b != nullptr) v += static_cast<AccT>(b[i * bs]);
                sumSq += v * v;
            }
            const AccT inv = AccT(1) / math::sd_sqrt<AccT, AccT>(sumSq / AccT(rowLen) + AccT(epsilon));
            for (LongType i = 0; i < rowLen; ++i) {
                AccT v = static_cast<AccT>(x[xo + i * xs]) + static_cast<AccT>(s[so + i * ss]);
                if (b != nullptr) v += static_cast<AccT>(b[i * bs]);
                z[zo + i * zs] = static_cast<T>(v * inv * static_cast<AccT>(g[i * gs]));
                if (h != nullptr) h[ho + i * hs] = static_cast<T>(v);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
}

void skipRmsNorm(LaunchContext* context, NDArray* input, NDArray* skip, NDArray* gamma,
                 NDArray* bias, NDArray* output, NDArray* hiddenOut, double epsilon) {
    if (input->isEmpty()) return;
    NDArray::preparePrimaryUse({output, hiddenOut}, {input, skip, gamma, bias});
    BUILD_DOUBLE_SELECTOR(input->dataType(), gamma->dataType(), skipRmsNorm_,
                          (input, skip, gamma, bias, output, hiddenOut, epsilon),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES);
    NDArray::registerPrimaryUse({output, hiddenOut}, {input, skip, gamma, bias});
}
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
