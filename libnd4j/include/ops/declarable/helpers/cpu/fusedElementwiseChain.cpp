/* ******************************************************************************
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

//
// @author Adam Gibson
//
// CPU implementation of the fused element-wise chain kernel.
//

#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <array/NDArray.h>
#include <execution/Threads.h>
#include <math/templatemath.h>
#include <cmath>
#include <algorithm>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static T applyOp(T val, FusedElemOp op, T secondaryVal, T clipMinVal, T clipMaxVal) {
    switch (op) {
        // Binary ops
        case FUSED_ADD:       return val + secondaryVal;
        case FUSED_SUB:       return val - secondaryVal;
        case FUSED_MUL:       return val * secondaryVal;
        case FUSED_DIV:       return secondaryVal != T(0) ? val / secondaryVal : T(0);

        // Unary ops
        case FUSED_RELU:      return val > T(0) ? val : T(0);
        case FUSED_SIGMOID:   return T(1) / (T(1) + std::exp(-val));
        case FUSED_TANH:      return std::tanh(val);
        case FUSED_GELU: {
            // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            T c = T(0.7978845608); // sqrt(2/pi)
            T inner = c * (val + T(0.044715) * val * val * val);
            return T(0.5) * val * (T(1) + std::tanh(inner));
        }
        case FUSED_EXP:       return std::exp(val);
        case FUSED_LOG:       return val > T(0) ? T(std::log(val)) : T(-1e38);
        case FUSED_ABS:       return sd::math::sd_abs<T, T>(val);
        case FUSED_NEG:       return -val;
        case FUSED_SQUARE:    return val * val;
        case FUSED_SQRT:      return val >= T(0) ? T(std::sqrt(val)) : T(0);
        case FUSED_SWISH:     return val / (T(1) + std::exp(-val)); // x * sigmoid(x)
        case FUSED_SILU:      return val / (T(1) + std::exp(-val)); // Same as swish
        case FUSED_MISH: {
            T sp = std::log(T(1) + std::exp(val)); // softplus
            return val * std::tanh(sp);
        }

        // Parameterized ops
        case FUSED_CLIP:      return std::min(std::max(val, clipMinVal), clipMaxVal);
        case FUSED_LEAKY_RELU: return val >= T(0) ? val : val * secondaryVal;

        default:              return val;
    }
}

template <typename T>
static void fusedChainImpl(
        NDArray* inputArr, NDArray* outputArr, sd::LongType length,
        const FusedElemOp* ops, int numOps,
        NDArray** secondaryInputs,
        const double* clipMin, const double* clipMax) {

    const T* input = inputArr->bufferAsT<T>();
    T* output = outputArr->bufferAsT<T>();
    T clipMinVal = clipMin ? static_cast<T>(*clipMin) : T(0);
    T clipMaxVal = clipMax ? static_cast<T>(*clipMax) : T(0);

    auto func = PRAGMA_THREADS_FOR {
        for (auto idx = start; idx < stop; idx++) {
            T val = input[idx];
            for (int op = 0; op < numOps; op++) {
                T secondary = T(0);
                if (isBinaryFusedOp(ops[op]) && secondaryInputs != nullptr && secondaryInputs[op] != nullptr) {
                    sd::LongType secLen = secondaryInputs[op]->lengthOf();
                    sd::LongType secIdx = secLen == 1 ? 0 : (idx % secLen);
                    secondary = secondaryInputs[op]->e<T>(secIdx);
                }
                val = applyOp(val, ops[op], secondary, clipMinVal, clipMaxVal);
            }
            output[idx] = val;
        }
    };

    samediff::Threads::parallel_for(func, 0, length);
}

void fusedElementwiseChain(
        NDArray* input,
        NDArray* output,
        const FusedElemOp* ops,
        int numOps,
        NDArray** secondaryInputs,
        const double* clipMin,
        const double* clipMax,
        LaunchContext* context) {

    if (input == nullptr || output == nullptr || ops == nullptr || numOps <= 0) return;
    if (numOps > 8) {
        numOps = 8;
    }

    auto xType = input->dataType();
    sd::LongType length = input->lengthOf();

    BUILD_SINGLE_SELECTOR(xType, fusedChainImpl,
        (input, output, length,
         ops, numOps, secondaryInputs, clipMin, clipMax),
        SD_COMMON_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
