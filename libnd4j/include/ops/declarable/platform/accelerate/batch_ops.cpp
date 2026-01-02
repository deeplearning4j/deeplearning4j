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
// Apple Accelerate framework - Batch operations
// Batched GEMM and other batched linear algebra operations
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE

namespace sd {
namespace ops {
namespace platforms {

//==============================================================================
// BATCHED_GEMM - Batched matrix multiplication using vecLib
// Computes C[i] = alpha * A[i] @ B[i] + beta * C[i] for each batch
//==============================================================================

PLATFORM_IMPL(batched_gemm, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);  // Shape: [batch, M, K]
    auto b = INPUT_VARIABLE(1);  // Shape: [batch, K, N]
    auto c = OUTPUT_VARIABLE(0); // Shape: [batch, M, N]

    bool transA = block.getBArguments()->size() > 0 ? B_ARG(0) : false;
    bool transB = block.getBArguments()->size() > 1 ? B_ARG(1) : false;
    float alpha = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 1.0f;
    float beta = block.getTArguments()->size() > 1 ? static_cast<float>(T_ARG(1)) : 0.0f;

    const int batchSize = static_cast<int>(a->sizeAt(0));
    const int M = transA ? static_cast<int>(a->sizeAt(2)) : static_cast<int>(a->sizeAt(1));
    const int K = transA ? static_cast<int>(a->sizeAt(1)) : static_cast<int>(a->sizeAt(2));
    const int N = transB ? static_cast<int>(b->sizeAt(1)) : static_cast<int>(b->sizeAt(2));

    const CBLAS_TRANSPOSE transAFlag = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE transBFlag = transB ? CblasTrans : CblasNoTrans;

    // Leading dimensions
    const int lda = transA ? M : K;
    const int ldb = transB ? K : N;
    const int ldc = N;

    // Stride between batches
    const sd::LongType strideA = a->sizeAt(1) * a->sizeAt(2);
    const sd::LongType strideB = b->sizeAt(1) * b->sizeAt(2);
    const sd::LongType strideC = c->sizeAt(1) * c->sizeAt(2);

    if (a->dataType() == DataType::FLOAT32) {
        const float* aBuf = a->bufferAsT<float>();
        const float* bBuf = b->bufferAsT<float>();
        float* cBuf = c->bufferAsT<float>();

        // Process each batch
        for (int batch = 0; batch < batchSize; batch++) {
            cblas_sgemm(CblasRowMajor, transAFlag, transBFlag,
                        M, N, K,
                        alpha,
                        aBuf + batch * strideA, lda,
                        bBuf + batch * strideB, ldb,
                        beta,
                        cBuf + batch * strideC, ldc);
        }
    } else if (a->dataType() == DataType::DOUBLE) {
        const double* aBuf = a->bufferAsT<double>();
        const double* bBuf = b->bufferAsT<double>();
        double* cBuf = c->bufferAsT<double>();
        double alphaD = static_cast<double>(alpha);
        double betaD = static_cast<double>(beta);

        for (int batch = 0; batch < batchSize; batch++) {
            cblas_dgemm(CblasRowMajor, transAFlag, transBFlag,
                        M, N, K,
                        alphaD,
                        aBuf + batch * strideA, lda,
                        bBuf + batch * strideB, ldb,
                        betaD,
                        cBuf + batch * strideC, ldc);
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(batched_gemm, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto c = OUTPUT_VARIABLE(0);

    Requirements reqs("BATCHED_GEMM ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, a, c);
    reqs.expectTrue(accelerateUtils::isContiguous(*a), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*b), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*c), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(a->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(a->rankOf() == 3, RANK_MSG_INPUT0);
    reqs.expectTrue(b->rankOf() == 3, RANK_MSG_INPUT1);

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
