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
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/rms_norm.h>
#if NOT_EXCLUDED(OP_rms_norm)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void rmsNorm_(NDArray* input, NDArray* gamma, NDArray* output, float epsilon) {
    const LongType numRows = input->lengthOf() / input->sizeAt(-1);
    const LongType rowLen = input->sizeAt(-1);

    // Use actual strides so non-contiguous views (e.g. from permute) are handled correctly.
    // For a permuted [1, 576, 768] view of a [1, 768, 576] base array the row stride is
    // input->strideAt(-2) (may be 1, not rowLen) and element stride input->strideAt(-1)
    // (may be 576, not 1).  Assuming row * rowLen offset is WRONG for such views.
    const LongType rowStride = input->rankOf() >= 2 ? input->strideAt(-2) : rowLen;
    const LongType elemStride = input->strideAt(-1);
    const LongType outRowStride = output->rankOf() >= 2 ? output->strideAt(-2) : rowLen;
    const LongType outElemStride = output->strideAt(-1);
    // gamma is always 1D contiguous
    const LongType gammaElemStride = (gamma != nullptr) ? gamma->strideAt(0) : 1;

    const T* x = input->bufferAsT<T>();
    T* z = output->bufferAsT<T>();
    const T* g = gamma != nullptr ? gamma->bufferAsT<T>() : nullptr;

    // Use double for accumulation when T is double to preserve precision.
    // For float16/bfloat16, accumulate in float to avoid FP16 overflow (65504 limit).
    using AccT = typename std::conditional<std::is_same<T, double>::value, double, float>::type;

    // Fast path: both input and output are contiguous (stride-1 in last dim, row stride = rowLen).
    // This is the common case and allows SIMD.
    const bool inputContig  = (elemStride == 1) && (rowStride == rowLen);
    const bool outputContig = (outElemStride == 1) && (outRowStride == rowLen);

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const LongType xOff = row * rowStride;
            const LongType zOff = row * outRowStride;

            // Accumulate sum-of-squares in AccT (float for FP16/BF16, double for FP64)
            AccT sumSq = static_cast<AccT>(0);
            if (inputContig) {
                const T* xRow = x + xOff;
                for (LongType i = 0; i < rowLen; ++i) {
                    AccT val = static_cast<AccT>(xRow[i]);
                    sumSq += val * val;
                }
            } else {
                for (LongType i = 0; i < rowLen; ++i) {
                    AccT val = static_cast<AccT>(x[xOff + i * elemStride]);
                    sumSq += val * val;
                }
            }
            const AccT invRms = static_cast<AccT>(1) /
                sd::math::sd_sqrt<AccT, AccT>(sumSq / static_cast<AccT>(rowLen) +
                                              static_cast<AccT>(epsilon));

            if (inputContig && outputContig) {
                // Both contiguous: use SIMD
                const T* xRow = x + xOff;
                T* zRow = z + zOff;
                if (g != nullptr) {
                    PRAGMA_OMP_SIMD
                    for (LongType i = 0; i < rowLen; ++i) {
                        zRow[i] = static_cast<T>(static_cast<AccT>(xRow[i]) * invRms *
                                                 static_cast<AccT>(g[i * gammaElemStride]));
                    }
                } else {
                    PRAGMA_OMP_SIMD
                    for (LongType i = 0; i < rowLen; ++i) {
                        zRow[i] = static_cast<T>(static_cast<AccT>(xRow[i]) * invRms);
                    }
                }
            } else {
                // Non-contiguous: use strided access
                if (g != nullptr) {
                    for (LongType i = 0; i < rowLen; ++i) {
                        z[zOff + i * outElemStride] = static_cast<T>(
                            static_cast<AccT>(x[xOff + i * elemStride]) * invRms *
                            static_cast<AccT>(g[i * gammaElemStride]));
                    }
                } else {
                    for (LongType i = 0; i < rowLen; ++i) {
                        z[zOff + i * outElemStride] = static_cast<T>(
                            static_cast<AccT>(x[xOff + i * elemStride]) * invRms);
                    }
                }
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
}

void rmsNorm(LaunchContext* context, NDArray* input, NDArray* gamma, NDArray* output, float epsilon) {
    NDArray::preparePrimaryUse({output}, {input, gamma});

    // Handle mixed-type gamma: cast gamma to match input dtype on CPU
    // (CUDA uses dual-type kernel templates instead)
    NDArray* gammaToUse = gamma;
    NDArray* gammaCast = nullptr;
    if (gamma != nullptr && gamma->dataType() != input->dataType()) {
        gammaCast = gamma->cast(input->dataType());
        gammaToUse = gammaCast;
    }

    BUILD_SINGLE_SELECTOR(input->dataType(), rmsNorm_, (input, gammaToUse, output, epsilon), SD_FLOAT_TYPES);

    if (gammaCast != nullptr) delete gammaCast;

    NDArray::registerPrimaryUse({output}, {input, gamma});
}

///////////////////////////////////////////////////////////////////////////////
// Fused RMSNorm + Linear: output = matmul(rmsNorm(input, gamma, eps), weight)
//
// Delegates normalization to the existing rmsNorm kernel and matmul to
// MmulHelper::mmul.  Everything runs in FLOAT32 to avoid HALF matmul
// producing zeros on CPUs without AMX-FP16 (e.g. AMD Ryzen).
///////////////////////////////////////////////////////////////////////////////
template <typename T>
static void rmsNormLinear_(NDArray* input, NDArray* gamma, NDArray* weight,
                            NDArray* output, float epsilon) {
    // 1. Normalize into a FLOAT32 buffer via the existing rmsNorm kernel
    auto shapeVec = *input->getShapeAsVector();
    NDArray normalized(input->ordering(), shapeVec, DataType::FLOAT32);

    // Cast input and gamma to FLOAT32 so rmsNorm_ runs the float path
    NDArray* inputF32 = input;
    NDArray* inputCasted = nullptr;
    if (input->dataType() != DataType::FLOAT32) {
        inputCasted = input->cast(DataType::FLOAT32);
        inputF32 = inputCasted;
    }

    NDArray* gammaF32 = gamma;
    NDArray* gammaCasted = nullptr;
    if (gamma != nullptr && gamma->dataType() != DataType::FLOAT32) {
        gammaCasted = gamma->cast(DataType::FLOAT32);
        gammaF32 = gammaCasted;
    }

    rmsNorm_<float>(inputF32, gammaF32, &normalized, epsilon);

    delete inputCasted;
    delete gammaCasted;

    // 2. Cast weight to FLOAT32 if needed
    NDArray* wF32 = weight;
    NDArray* wCasted = nullptr;
    if (weight->dataType() != DataType::FLOAT32) {
        wCasted = weight->cast(DataType::FLOAT32);
        wF32 = wCasted;
    }

    // 3. Matmul in FP32: normalized [M, K] @ weight [K, N] -> output [M, N]
    if (output->dataType() == DataType::FLOAT32) {
        MmulHelper::mmul(&normalized, wF32, output, 1.0, 0.0);
    } else {
        auto outShape = *output->getShapeAsVector();
        NDArray outF32(output->ordering(), outShape, DataType::FLOAT32);
        MmulHelper::mmul(&normalized, wF32, &outF32, 1.0, 0.0);
        output->assign(&outF32);
    }

    delete wCasted;
}

void rmsNormLinear(LaunchContext* context, NDArray* input, NDArray* gamma,
                    NDArray* weight, NDArray* output, float epsilon) {
    NDArray::preparePrimaryUse({output}, {input, gamma, weight});

    BUILD_SINGLE_SELECTOR(input->dataType(), rmsNormLinear_,
                           (input, gamma, weight, output, epsilon), SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output}, {input, gamma, weight});
}

///////////////////////////////////////////////////////////////////////////////
// Fused Skip (Residual Add) + RMS Normalization
///////////////////////////////////////////////////////////////////////////////
template <typename T>
static void skipRmsNorm_(NDArray* input, NDArray* skip, NDArray* gamma, NDArray* bias,
                          NDArray* output, NDArray* hiddenOut, float epsilon) {
    const LongType numRows = input->lengthOf() / input->sizeAt(-1);
    const LongType rowLen = input->sizeAt(-1);

    // Use actual strides to handle non-contiguous views correctly.
    const LongType xRowStride  = input->rankOf() >= 2 ? input->strideAt(-2) : rowLen;
    const LongType xElemStride = input->strideAt(-1);
    const LongType sRowStride  = skip->rankOf() >= 2 ? skip->strideAt(-2) : rowLen;
    const LongType sElemStride = skip->strideAt(-1);
    const LongType zRowStride  = output->rankOf() >= 2 ? output->strideAt(-2) : rowLen;
    const LongType zElemStride = output->strideAt(-1);
    const LongType hRowStride  = (hiddenOut != nullptr && hiddenOut->rankOf() >= 2) ? hiddenOut->strideAt(-2) : rowLen;
    const LongType hElemStride = hiddenOut != nullptr ? hiddenOut->strideAt(-1) : 1;
    // gamma and bias are always 1D contiguous
    const LongType gElemStride = gamma->strideAt(0);
    const LongType bElemStride = bias != nullptr ? bias->strideAt(0) : 1;

    const T* x = input->bufferAsT<T>();
    const T* s = skip->bufferAsT<T>();
    const T* g = gamma->bufferAsT<T>();
    const T* b = bias != nullptr ? bias->bufferAsT<T>() : nullptr;
    T* z = output->bufferAsT<T>();
    T* h = hiddenOut != nullptr ? hiddenOut->bufferAsT<T>() : nullptr;

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const LongType xOff = row * xRowStride;
            const LongType sOff = row * sRowStride;
            const LongType zOff = row * zRowStride;
            const LongType hOff = row * hRowStride;

            // Pass 1: compute hidden = input + skip [+ bias], accumulate sum of squares in float
            float sumSq = 0.0f;
            for (LongType i = 0; i < rowLen; ++i) {
                float val = static_cast<float>(x[xOff + i * xElemStride])
                          + static_cast<float>(s[sOff + i * sElemStride]);
                if (b != nullptr) val += static_cast<float>(b[i * bElemStride]);
                if (h != nullptr) h[hOff + i * hElemStride] = static_cast<T>(val);
                sumSq += val * val;
            }
            const float invRms = 1.0f / sd::math::sd_sqrt<float, float>(sumSq / static_cast<float>(rowLen) + epsilon);

            // Pass 2: normalize and scale
            for (LongType i = 0; i < rowLen; ++i) {
                float val = static_cast<float>(x[xOff + i * xElemStride])
                          + static_cast<float>(s[sOff + i * sElemStride]);
                if (b != nullptr) val += static_cast<float>(b[i * bElemStride]);
                z[zOff + i * zElemStride] = static_cast<T>(val * invRms * static_cast<float>(g[i * gElemStride]));
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
}

void skipRmsNorm(LaunchContext* context, NDArray* input, NDArray* skip, NDArray* gamma,
                  NDArray* bias, NDArray* output, NDArray* hiddenOut, float epsilon) {
    NDArray::preparePrimaryUse({output, hiddenOut}, {input, skip, gamma, bias});

    // Handle mixed-type gamma: cast gamma to match input dtype on CPU
    // (CUDA uses dual-type kernel templates instead)
    NDArray* gammaToUse = gamma;
    NDArray* gammaCast = nullptr;
    if (gamma != nullptr && gamma->dataType() != input->dataType()) {
        gammaCast = gamma->cast(input->dataType());
        gammaToUse = gammaCast;
    }

    BUILD_SINGLE_SELECTOR(input->dataType(), skipRmsNorm_,
                           (input, skip, gammaToUse, bias, output, hiddenOut, epsilon), SD_FLOAT_TYPES);

    if (gammaCast != nullptr) delete gammaCast;

    NDArray::registerPrimaryUse({output, hiddenOut}, {input, skip, gamma, bias});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
