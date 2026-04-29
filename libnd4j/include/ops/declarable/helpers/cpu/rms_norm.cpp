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

    const T* x = input->bufferAsT<T>();
    T* z = output->bufferAsT<T>();
    const T* g = gamma != nullptr ? gamma->bufferAsT<T>() : nullptr;

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * rowLen;
            T* zRow = z + row * rowLen;

            // Accumulate in float to avoid FP16 overflow (1024-dim sum-of-squares exceeds 65504)
            float sumSq = 0.0f;
            for (LongType i = 0; i < rowLen; ++i) {
                float val = static_cast<float>(xRow[i]);
                sumSq += val * val;
            }
            const float invRms = 1.0f / std::sqrt(sumSq / static_cast<float>(rowLen) + epsilon);

            if (g != nullptr) {
                PRAGMA_OMP_SIMD
                for (LongType i = 0; i < rowLen; ++i) {
                    zRow[i] = static_cast<T>(static_cast<float>(xRow[i]) * invRms * static_cast<float>(g[i]));
                }
            } else {
                PRAGMA_OMP_SIMD
                for (LongType i = 0; i < rowLen; ++i) {
                    zRow[i] = static_cast<T>(static_cast<float>(xRow[i]) * invRms);
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

    const T* x = input->bufferAsT<T>();
    const T* s = skip->bufferAsT<T>();
    const T* g = gamma->bufferAsT<T>();
    const T* b = bias != nullptr ? bias->bufferAsT<T>() : nullptr;
    T* z = output->bufferAsT<T>();
    T* h = hiddenOut != nullptr ? hiddenOut->bufferAsT<T>() : nullptr;

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * rowLen;
            const T* sRow = s + row * rowLen;
            T* zRow = z + row * rowLen;
            T* hRow = h != nullptr ? h + row * rowLen : nullptr;

            // Pass 1: compute hidden = input + skip [+ bias], accumulate sum of squares in float
            float sumSq = 0.0f;
            for (LongType i = 0; i < rowLen; ++i) {
                float val = static_cast<float>(xRow[i]) + static_cast<float>(sRow[i]);
                if (b != nullptr) val += static_cast<float>(b[i]);
                if (hRow != nullptr) hRow[i] = static_cast<T>(val);
                sumSq += val * val;
            }
            const float invRms = 1.0f / std::sqrt(sumSq / static_cast<float>(rowLen) + epsilon);

            // Pass 2: normalize and scale
            PRAGMA_OMP_SIMD
            for (LongType i = 0; i < rowLen; ++i) {
                float val = static_cast<float>(xRow[i]) + static_cast<float>(sRow[i]);
                if (b != nullptr) val += static_cast<float>(b[i]);
                zRow[i] = static_cast<T>(val * invRms * static_cast<float>(g[i]));
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
