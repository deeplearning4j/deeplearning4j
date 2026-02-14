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
// CUDA implementation of the fused element-wise chain kernel.
// Processes the entire chain of operations per-element in a single kernel,
// keeping intermediate values in registers instead of global memory.
//

#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
__device__ T deviceApplyOp(T val, FusedElemOp op, T secondaryVal, T clipMinVal, T clipMaxVal) {
    switch (op) {
        // Binary ops
        case FUSED_ADD:       return val + secondaryVal;
        case FUSED_SUB:       return val - secondaryVal;
        case FUSED_MUL:       return val * secondaryVal;
        case FUSED_DIV:       return secondaryVal != T(0) ? val / secondaryVal : T(0);

        // Unary ops
        case FUSED_RELU:      return val > T(0) ? val : T(0);
        case FUSED_SIGMOID:   return T(1) / (T(1) + __expf(-static_cast<float>(val)));
        case FUSED_TANH:      return static_cast<T>(tanhf(static_cast<float>(val)));
        case FUSED_GELU: {
            float x = static_cast<float>(val);
            float c = 0.7978845608f; // sqrt(2/pi)
            float inner = c * (x + 0.044715f * x * x * x);
            return static_cast<T>(0.5f * x * (1.0f + tanhf(inner)));
        }
        case FUSED_EXP:       return static_cast<T>(__expf(static_cast<float>(val)));
        case FUSED_LOG:       return val > T(0) ? static_cast<T>(__logf(static_cast<float>(val))) : T(-1e38);
        case FUSED_ABS:       return val >= T(0) ? val : -val;
        case FUSED_NEG:       return -val;
        case FUSED_SQUARE:    return val * val;
        case FUSED_SQRT:      return val >= T(0) ? static_cast<T>(sqrtf(static_cast<float>(val))) : T(0);
        case FUSED_SWISH: {
            float sig = 1.0f / (1.0f + __expf(-static_cast<float>(val)));
            return static_cast<T>(static_cast<float>(val) * sig);
        }
        case FUSED_SILU: {
            float sig = 1.0f / (1.0f + __expf(-static_cast<float>(val)));
            return static_cast<T>(static_cast<float>(val) * sig);
        }
        case FUSED_MISH: {
            float x = static_cast<float>(val);
            float sp = __logf(1.0f + __expf(x)); // softplus
            return static_cast<T>(x * tanhf(sp));
        }

        // Parameterized ops
        case FUSED_CLIP:      return val < clipMinVal ? clipMinVal : (val > clipMaxVal ? clipMaxVal : val);
        case FUSED_LEAKY_RELU: return val >= T(0) ? val : val * secondaryVal;

        default:              return val;
    }
}

/**
 * CUDA kernel: process the entire chain per-element.
 *
 * Each thread handles one element, applying all ops in sequence.
 * Intermediate values stay in registers — no global memory traffic
 * between ops. This is O(1) global mem reads/writes regardless of
 * chain length, vs O(N) for N separate kernels.
 *
 * Max 8 secondary input pointers are passed via constant args to avoid
 * extra global memory loads. FusedElemOp codes are stored in shared memory
 * for fast access (they fit in one cache line).
 */
template <typename T>
__global__ void fusedElemKernel(
        const T* __restrict__ input,
        T* __restrict__ output,
        sd::LongType length,
        const FusedElemOp* __restrict__ ops,
        int numOps,
        const T* __restrict__ sec0, sd::LongType secLen0,
        const T* __restrict__ sec1, sd::LongType secLen1,
        const T* __restrict__ sec2, sd::LongType secLen2,
        const T* __restrict__ sec3, sd::LongType secLen3,
        const T* __restrict__ sec4, sd::LongType secLen4,
        const T* __restrict__ sec5, sd::LongType secLen5,
        const T* __restrict__ sec6, sd::LongType secLen6,
        const T* __restrict__ sec7, sd::LongType secLen7,
        T clipMinVal, T clipMaxVal) {

    // Load op codes into shared memory for fast access
    __shared__ FusedElemOp sharedOps[8];
    if (threadIdx.x < numOps && threadIdx.x < 8) {
        sharedOps[threadIdx.x] = ops[threadIdx.x];
    }
    __syncthreads();

    // Store secondary input pointers and lengths in arrays for indexed access
    const T* secPtrs[8] = {sec0, sec1, sec2, sec3, sec4, sec5, sec6, sec7};
    sd::LongType secLens[8] = {secLen0, secLen1, secLen2, secLen3, secLen4, secLen5, secLen6, secLen7};

    auto idx = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= length) return;

    T val = input[idx];

    for (int op = 0; op < numOps; op++) {
        T secondary = T(0);
        if (isBinaryFusedOp(sharedOps[op]) && secPtrs[op] != nullptr) {
            sd::LongType secIdx = secLens[op] == 1 ? 0 : (idx % secLens[op]);
            secondary = secPtrs[op][secIdx];
        }
        val = deviceApplyOp(val, sharedOps[op], secondary, clipMinVal, clipMaxVal);
    }

    output[idx] = val;
}

template <typename T>
static void fusedChainCudaImpl(
        NDArray* input, NDArray* output,
        const FusedElemOp* ops, int numOps,
        NDArray** secondaryInputs,
        const double* clipMin, const double* clipMax,
        LaunchContext* context) {

    sd::LongType length = input->lengthOf();
    if (length == 0) return;

    auto stream = context->getCudaStream();

    // Prepare secondary input pointers (up to 8)
    const T* secPtrs[8] = {nullptr};
    sd::LongType secLens[8] = {0};

    if (secondaryInputs != nullptr) {
        for (int i = 0; i < numOps && i < 8; i++) {
            if (secondaryInputs[i] != nullptr) {
                secPtrs[i] = reinterpret_cast<const T*>(secondaryInputs[i]->specialBuffer());
                secLens[i] = secondaryInputs[i]->lengthOf();
            }
        }
    }

    T clipMinVal = clipMin ? static_cast<T>(*clipMin) : T(0);
    T clipMaxVal = clipMax ? static_cast<T>(*clipMax) : T(0);

    // Copy op codes to device memory (stream-ordered alloc)
    FusedElemOp* deviceOps = nullptr;
    cudaMallocAsync(&deviceOps, numOps * sizeof(FusedElemOp), *stream);
    cudaMemcpyAsync(deviceOps, ops, numOps * sizeof(FusedElemOp),
                    cudaMemcpyHostToDevice, *stream);

    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;

    fusedElemKernel<T><<<gridSize, blockSize, 0, *stream>>>(
            reinterpret_cast<const T*>(input->specialBuffer()),
            reinterpret_cast<T*>(output->specialBuffer()),
            length, deviceOps, numOps,
            secPtrs[0], secLens[0], secPtrs[1], secLens[1],
            secPtrs[2], secLens[2], secPtrs[3], secLens[3],
            secPtrs[4], secLens[4], secPtrs[5], secLens[5],
            secPtrs[6], secLens[6], secPtrs[7], secLens[7],
            clipMinVal, clipMaxVal);

    // Free device op codes after kernel completes
    cudaFreeAsync(deviceOps, *stream);
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
        // Chain too long — fall back to splitting into two calls
        // (not implemented yet, just process first 8)
        numOps = 8;
    }

    auto xType = input->dataType();

    // Dispatch based on data type
    if (xType == DataType::FLOAT32) {
        fusedChainCudaImpl<float>(input, output, ops, numOps, secondaryInputs, clipMin, clipMax, context);
    } else if (xType == DataType::DOUBLE) {
        fusedChainCudaImpl<double>(input, output, ops, numOps, secondaryInputs, clipMin, clipMax, context);
    } else if (xType == DataType::HALF) {
        fusedChainCudaImpl<float16>(input, output, ops, numOps, secondaryInputs, clipMin, clipMax, context);
    } else if (xType == DataType::BFLOAT16) {
        fusedChainCudaImpl<bfloat16>(input, output, ops, numOps, secondaryInputs, clipMin, clipMax, context);
    } else {
        // Unsupported type — should not happen for element-wise chains
        sd_printf("fusedElementwiseChain: unsupported dtype %d\n", static_cast<int>(xType));
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
