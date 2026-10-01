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

//
// INT8 scaled GEMM — CUDA implementation.
// Uses cublasLtMatmul with CUBLAS_COMPUTE_32I for native INT8 tensor cores.
//

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <types/float16.h>
#include <ops/declarable/helpers/int8_gemm.h>
#include <helpers/cublasHelper.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Dequantize + apply bias kernel
// output_fp = int32_accum * scaleA * scaleB + bias
// Fused into a single kernel to avoid extra global memory passes.
//////////////////////////////////////////////////////////////////////////////
template <typename OutT>
SD_KERNEL void dequantizeInt32Kernel(
    const int32_t* __restrict__ accumulator,  // [M, N] INT32 GEMM output
    const float* __restrict__ scaleA,         // [M] or [1] per-token or per-tensor
    const float* __restrict__ scaleB,         // [N] or [1] per-channel or per-tensor
    const float* __restrict__ bias,           // [N] or nullptr
    OutT* __restrict__ output,                // [M, N]
    const LongType M,
    const LongType N,
    const bool perTokenA,                     // true if scaleA is [M], false if [1]
    const bool perChannelB) {                 // true if scaleB is [N], false if [1]

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    const LongType row = idx / N;
    const LongType col = idx % N;

    float sA = perTokenA ? scaleA[row] : scaleA[0];
    float sB = perChannelB ? scaleB[col] : scaleB[0];

    float val = static_cast<float>(accumulator[idx]) * sA * sB;

    if (bias != nullptr) {
        val += bias[col];
    }

    output[idx] = static_cast<OutT>(val);
}

//////////////////////////////////////////////////////////////////////////////
// Public: int8ScaledGemm
// C_int32 = A_int8 * B_int8 via cublasLt, then dequantize to output type.
//////////////////////////////////////////////////////////////////////////////
void int8ScaledGemm(LaunchContext* context,
                     NDArray* A,
                     NDArray* B,
                     NDArray* scaleA,
                     NDArray* scaleB,
                     NDArray* output,
                     NDArray* bias) {
    const LongType M = A->sizeAt(0);
    const LongType K = A->sizeAt(1);
    const LongType N = B->sizeAt(1);

    NDArray::prepareSpecialUse({output}, {A, B, scaleA, scaleB});
    if (bias != nullptr) NDArray::prepareSpecialUse({}, {bias});

    auto stream = context->getCudaStream();

    // Allocate INT32 accumulator
    std::vector<LongType> accShape = {M, N};
    auto accumulator = NDArrayFactory::create_('c', accShape, INT32, context);
    NDArray::prepareSpecialUse({accumulator}, {});

    // cublasLt setup
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);

    // Layout: row-major for A [M, K], B [K, N], C [M, N]
    // cublasLt uses column-major by default. For row-major:
    //   C_row = A_row * B_row is equivalent to C^T_col = B^T_col * A^T_col
    //   So we swap A and B, and use transB=T, transA=N
    cublasOperation_t transA = CUBLAS_OP_T;  // B^T in col-major
    cublasOperation_t transB = CUBLAS_OP_N;  // A in col-major
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                    &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                    &transB, sizeof(transB));

    // Matrix layouts (column-major for cublasLt)
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    // B^T: [N, K] col-major → leading dim = N
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, N, K, N);
    // A: [K, M] col-major → leading dim = K
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, K, M, K);
    // C: [N, M] col-major → leading dim = N
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, N, M, N);

    // Alpha/beta scalars
    int32_t alpha = 1;
    int32_t beta = 0;

    // Preference: use default algorithm
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    size_t workspaceSize = 4 * 1024 * 1024;  // 4MB
    cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    cublasLtMatmulHeuristicResult_t heurResult;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, layoutA, layoutB,
                                    layoutC, layoutC, preference, 1,
                                    &heurResult, &returnedResults);

    void* workspace = nullptr;
    int int8GemmDeviceId = context->getDeviceID();
    if (heurResult.workspaceSize > 0) {
        workspace = memory::CudaMemoryPool::getInstance().allocate(
            heurResult.workspaceSize, int8GemmDeviceId, *stream);
    }

    // Execute INT8 GEMM: swap A/B for row-major
    auto status = cublasLtMatmul(ltHandle, matmulDesc,
                                  &alpha,
                                  B->specialBuffer(), layoutA,  // B^T
                                  A->specialBuffer(), layoutB,  // A
                                  &beta,
                                  accumulator->specialBuffer(), layoutC,
                                  accumulator->specialBuffer(), layoutC,
                                  (returnedResults > 0) ? &heurResult.algo : nullptr,
                                  workspace, heurResult.workspaceSize,
                                  *stream);

    // Cleanup cublasLt objects
    if (workspace) memory::CudaMemoryPool::getInstance().free(workspace, int8GemmDeviceId, *stream);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(ltHandle);

    NDArray::registerSpecialUse({accumulator}, {A, B});

    // Dequantize INT32 → output type
    bool perTokenA = (scaleA->lengthOf() > 1);
    bool perChannelB = (scaleB->lengthOf() > 1);
    const float* biasPtr = (bias != nullptr) ?
        reinterpret_cast<const float*>(bias->specialBuffer()) : nullptr;

    int threads = 256;
    int blocks = static_cast<int>((M * N + threads - 1) / threads);

    auto outDtype = output->dataType();
    if (outDtype == DataType::FLOAT32) {
        dequantizeInt32Kernel<float><<<blocks, threads, 0, *stream>>>(
            reinterpret_cast<const int32_t*>(accumulator->specialBuffer()),
            reinterpret_cast<const float*>(scaleA->specialBuffer()),
            reinterpret_cast<const float*>(scaleB->specialBuffer()),
            biasPtr,
            reinterpret_cast<float*>(output->specialBuffer()),
            M, N, perTokenA, perChannelB);
    } else if (outDtype == DataType::HALF) {
        dequantizeInt32Kernel<float16><<<blocks, threads, 0, *stream>>>(
            reinterpret_cast<const int32_t*>(accumulator->specialBuffer()),
            reinterpret_cast<const float*>(scaleA->specialBuffer()),
            reinterpret_cast<const float*>(scaleB->specialBuffer()),
            biasPtr,
            reinterpret_cast<float16*>(output->specialBuffer()),
            M, N, perTokenA, perChannelB);
    } else {
        THROW_EXCEPTION("int8ScaledGemm: unsupported output type (use FLOAT32 or HALF)");
    }

    delete accumulator;

    DebugHelper::checkGlobalErrorCode("int8ScaledGemm failed");
    NDArray::registerSpecialUse({output}, {scaleA, scaleB});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
