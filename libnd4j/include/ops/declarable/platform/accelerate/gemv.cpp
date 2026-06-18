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
// Apple Accelerate framework - Matrix-Vector multiplication via vecLib BLAS
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_ACCELERATE

/**
 * Perform matrix-vector multiplication using Accelerate's CBLAS
 * y = alpha * op(A) * x + beta * y
 *
 * Where:
 *   A is an M x N matrix
 *   x is a vector of length N (or M if transposed)
 *   y is a vector of length M (or N if transposed)
 */
static void gemvAccelerate(const NDArray* a, const NDArray* x, NDArray* y,
                            bool transA, double alpha, double beta) {
    // Get dimensions
    const int M = static_cast<int>(a->sizeAt(0));
    const int N = static_cast<int>(a->sizeAt(1));
    const int lda = static_cast<int>(a->strideAt(0));

    // Vector strides
    const int incX = static_cast<int>(x->strideAt(0));
    const int incY = static_cast<int>(y->strideAt(0));

    CBLAS_TRANSPOSE transAOp = transA ? CblasTrans : CblasNoTrans;

    if (a->dataType() == DataType::FLOAT32) {
        cblas_sgemv(CblasRowMajor, transAOp,
                    M, N,
                    static_cast<float>(alpha),
                    a->bufferAsT<float>(), lda,
                    x->bufferAsT<float>(), incX,
                    static_cast<float>(beta),
                    y->bufferAsT<float>(), incY);
    } else if (a->dataType() == DataType::DOUBLE) {
        cblas_dgemv(CblasRowMajor, transAOp,
                    M, N,
                    alpha,
                    a->bufferAsT<double>(), lda,
                    x->bufferAsT<double>(), incX,
                    beta,
                    y->bufferAsT<double>(), incY);
    }
}

PLATFORM_IMPL(gemv, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);  // Matrix [M, N]
    auto x = INPUT_VARIABLE(1);  // Vector [N] or [M] if transposed
    auto y = OUTPUT_VARIABLE(0); // Vector [M] or [N] if transposed

    // Get transpose flag from integer arguments
    int transA = 0;
    if (block.getIArguments()->size() > 0)
        transA = INT_ARG(0);

    // Get alpha/beta from T arguments (default to 1.0 and 0.0)
    double alpha = 1.0;
    double beta = 0.0;

    if (block.getTArguments()->size() > 0)
        alpha = T_ARG(0);
    if (block.getTArguments()->size() > 1)
        beta = T_ARG(1);

    gemvAccelerate(a, x, y, transA != 0, alpha, beta);

    return sd::Status::OK;
}

PLATFORM_CHECK(gemv, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);

    Requirements req("ACCELERATE GEMV OP");

    // Check if Accelerate is enabled
    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);

    // Check data type support (float32 and float64 only for BLAS)
    req.expectTrue(a->dataType() == DataType::FLOAT32 || a->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported by Accelerate BLAS");

    // Data types must match
    req.expectTrue(a->dataType() == x->dataType(),
                   "Matrix and vector must have the same data type");

    // Matrix must be 2D
    req.expectTrue(a->rankOf() == 2, "Matrix A must be 2D");

    // Vector must be 1D
    req.expectTrue(x->rankOf() == 1, "Vector x must be 1D");

    // Arrays must be contiguous
    req.expectTrue(accelerateUtils::isContiguous(*a),
                   "Matrix A must be contiguous (ews=1)");

    // Arrays must not be empty
    req.expectFalse(a->isEmpty(), "Matrix A must not be empty");
    req.expectFalse(x->isEmpty(), "Vector x must not be empty");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_ACCELERATE

}  // namespace platforms
}  // namespace ops
}  // namespace sd
