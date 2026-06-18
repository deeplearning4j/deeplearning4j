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
// Apple Accelerate framework - Additional BLAS operations
// Uses vecLib for scal, ger (outer product), copy, swap, and other BLAS ops
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
// SCAL - Scalar multiplication: x = alpha * x
//==============================================================================

PLATFORM_IMPL(scal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    // Get alpha from T arguments
    double alpha = T_ARG(0);

    const int n = static_cast<int>(x->lengthOf());

    // Copy input to output if not in-place
    if (z != x) {
        z->assign(x);
    }

    if (z->dataType() == DataType::FLOAT32) {
        cblas_sscal(n, static_cast<float>(alpha), z->bufferAsT<float>(), 1);
    } else if (z->dataType() == DataType::DOUBLE) {
        cblas_dscal(n, alpha, z->bufferAsT<double>(), 1);
    }

    return Status::OK;
}

PLATFORM_CHECK(scal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("SCAL ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(block.getTArguments()->size() >= 1, "Need alpha T argument");

    return reqs.ok();
}

//==============================================================================
// GER - Outer product: A = alpha * x * y^T + A
//==============================================================================

PLATFORM_IMPL(ger, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);  // vector of length m
    auto y = INPUT_VARIABLE(1);  // vector of length n
    auto z = OUTPUT_VARIABLE(0); // matrix m x n

    double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

    const int m = static_cast<int>(x->lengthOf());
    const int n = static_cast<int>(y->lengthOf());

    // Initialize output to zero if needed
    z->nullify();

    if (x->dataType() == DataType::FLOAT32) {
        cblas_sger(CblasRowMajor, m, n,
                   static_cast<float>(alpha),
                   x->bufferAsT<float>(), 1,
                   y->bufferAsT<float>(), 1,
                   z->bufferAsT<float>(), n);
    } else if (x->dataType() == DataType::DOUBLE) {
        cblas_dger(CblasRowMajor, m, n,
                   alpha,
                   x->bufferAsT<double>(), 1,
                   y->bufferAsT<double>(), 1,
                   z->bufferAsT<double>(), n);
    }

    return Status::OK;
}

PLATFORM_CHECK(ger, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("GER ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->rankOf() == 1, RANK_MSG_INPUT0);
    reqs.expectTrue(y->rankOf() == 1, RANK_MSG_INPUT1);
    reqs.expectTrue(z->rankOf() == 2, RANK_MSG_OUTPUT0);

    return reqs.ok();
}

//==============================================================================
// COPY - Vector copy: y = x
//==============================================================================

PLATFORM_IMPL(copy, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        cblas_scopy(n, x->bufferAsT<float>(), 1, z->bufferAsT<float>(), 1);
    } else if (x->dataType() == DataType::DOUBLE) {
        cblas_dcopy(n, x->bufferAsT<double>(), 1, z->bufferAsT<double>(), 1);
    }

    return Status::OK;
}

PLATFORM_CHECK(copy, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("COPY ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SWAP - Vector swap: swap x and y
//==============================================================================

PLATFORM_IMPL(swap, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        cblas_sswap(n, x->bufferAsT<float>(), 1, y->bufferAsT<float>(), 1);
    } else if (x->dataType() == DataType::DOUBLE) {
        cblas_dswap(n, x->bufferAsT<double>(), 1, y->bufferAsT<double>(), 1);
    }

    return Status::OK;
}

PLATFORM_CHECK(swap, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements reqs("SWAP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->isSameShape(y), SHAPE_MSG_INPUT1);

    return reqs.ok();
}

//==============================================================================
// ROTG - Generate Givens rotation
//==============================================================================

PLATFORM_IMPL(rotg, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);  // scalar
    auto b = INPUT_VARIABLE(1);  // scalar
    auto c = OUTPUT_VARIABLE(0); // scalar (cosine)
    auto s = OUTPUT_VARIABLE(1); // scalar (sine)

    if (a->dataType() == DataType::FLOAT32) {
        float aVal = a->e<float>(0);
        float bVal = b->e<float>(0);
        float cVal, sVal;
        cblas_srotg(&aVal, &bVal, &cVal, &sVal);
        c->p(0, cVal);
        s->p(0, sVal);
    } else if (a->dataType() == DataType::DOUBLE) {
        double aVal = a->e<double>(0);
        double bVal = b->e<double>(0);
        double cVal, sVal;
        cblas_drotg(&aVal, &bVal, &cVal, &sVal);
        c->p(0, cVal);
        s->p(0, sVal);
    }

    return Status::OK;
}

PLATFORM_CHECK(rotg, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);

    Requirements reqs("ROTG ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, a, nullptr);
    reqs.expectTrue(a->isScalar(), "Input a must be scalar");
    reqs.expectTrue(b->isScalar(), "Input b must be scalar");
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(a->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ROT - Apply Givens rotation
//==============================================================================

PLATFORM_IMPL(rot, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto c = INPUT_VARIABLE(2);  // cosine scalar
    auto s = INPUT_VARIABLE(3);  // sine scalar

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        float cVal = c->e<float>(0);
        float sVal = s->e<float>(0);
        cblas_srot(n, x->bufferAsT<float>(), 1, y->bufferAsT<float>(), 1, cVal, sVal);
    } else if (x->dataType() == DataType::DOUBLE) {
        double cVal = c->e<double>(0);
        double sVal = s->e<double>(0);
        cblas_drot(n, x->bufferAsT<double>(), 1, y->bufferAsT<double>(), 1, cVal, sVal);
    }

    return Status::OK;
}

PLATFORM_CHECK(rot, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto c = INPUT_VARIABLE(2);
    auto s = INPUT_VARIABLE(3);

    Requirements reqs("ROT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(c->isScalar(), "c must be scalar");
    reqs.expectTrue(s->isScalar(), "s must be scalar");
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->isSameShape(y), SHAPE_MSG_INPUT1);

    return reqs.ok();
}

//==============================================================================
// SYMM - Symmetric matrix-matrix multiply
//==============================================================================

PLATFORM_IMPL(symm, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);  // symmetric matrix
    auto b = INPUT_VARIABLE(1);  // general matrix
    auto c = OUTPUT_VARIABLE(0); // output matrix

    double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
    double beta = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.0;

    // Get side (left or right) from I arguments
    bool leftSide = block.getIArguments()->size() > 0 ? (INT_ARG(0) == 0) : true;
    bool upperTriangle = block.getIArguments()->size() > 1 ? (INT_ARG(1) == 0) : true;

    const int m = static_cast<int>(c->sizeAt(0));
    const int n = static_cast<int>(c->sizeAt(1));
    const int lda = leftSide ? m : n;
    const int ldb = static_cast<int>(b->sizeAt(1));
    const int ldc = n;

    if (a->dataType() == DataType::FLOAT32) {
        cblas_ssymm(CblasRowMajor,
                    leftSide ? CblasLeft : CblasRight,
                    upperTriangle ? CblasUpper : CblasLower,
                    m, n,
                    static_cast<float>(alpha),
                    a->bufferAsT<float>(), lda,
                    b->bufferAsT<float>(), ldb,
                    static_cast<float>(beta),
                    c->bufferAsT<float>(), ldc);
    } else if (a->dataType() == DataType::DOUBLE) {
        cblas_dsymm(CblasRowMajor,
                    leftSide ? CblasLeft : CblasRight,
                    upperTriangle ? CblasUpper : CblasLower,
                    m, n,
                    alpha,
                    a->bufferAsT<double>(), lda,
                    b->bufferAsT<double>(), ldb,
                    beta,
                    c->bufferAsT<double>(), ldc);
    }

    return Status::OK;
}

PLATFORM_CHECK(symm, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto c = OUTPUT_VARIABLE(0);

    Requirements reqs("SYMM ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, a, c);
    reqs.expectTrue(accelerateUtils::isContiguous(*a), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*b), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*c), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(a->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(a->rankOf() == 2, RANK_MSG_INPUT0);
    reqs.expectTrue(b->rankOf() == 2, RANK_MSG_INPUT1);
    reqs.expectTrue(c->rankOf() == 2, RANK_MSG_OUTPUT0);

    return reqs.ok();
}

//==============================================================================
// SYRK - Symmetric rank-k update: C = alpha * A * A^T + beta * C
//==============================================================================

PLATFORM_IMPL(syrk, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto c = OUTPUT_VARIABLE(0);

    double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
    double beta = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.0;

    bool upperTriangle = block.getIArguments()->size() > 0 ? (INT_ARG(0) == 0) : true;
    bool transpose = block.getIArguments()->size() > 1 ? (INT_ARG(1) != 0) : false;

    const int n = static_cast<int>(c->sizeAt(0));
    const int k = transpose ? static_cast<int>(a->sizeAt(0)) : static_cast<int>(a->sizeAt(1));
    const int lda = transpose ? n : k;
    const int ldc = n;

    if (a->dataType() == DataType::FLOAT32) {
        cblas_ssyrk(CblasRowMajor,
                    upperTriangle ? CblasUpper : CblasLower,
                    transpose ? CblasTrans : CblasNoTrans,
                    n, k,
                    static_cast<float>(alpha),
                    a->bufferAsT<float>(), lda,
                    static_cast<float>(beta),
                    c->bufferAsT<float>(), ldc);
    } else if (a->dataType() == DataType::DOUBLE) {
        cblas_dsyrk(CblasRowMajor,
                    upperTriangle ? CblasUpper : CblasLower,
                    transpose ? CblasTrans : CblasNoTrans,
                    n, k,
                    alpha,
                    a->bufferAsT<double>(), lda,
                    beta,
                    c->bufferAsT<double>(), ldc);
    }

    return Status::OK;
}

PLATFORM_CHECK(syrk, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto c = OUTPUT_VARIABLE(0);

    Requirements reqs("SYRK ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, a, c);
    reqs.expectTrue(accelerateUtils::isContiguous(*a), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*c), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(a->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(a->rankOf() == 2, RANK_MSG_INPUT0);
    reqs.expectTrue(c->rankOf() == 2, RANK_MSG_OUTPUT0);
    // c must be square
    reqs.expectTrue(c->sizeAt(0) == c->sizeAt(1), "Output must be square matrix");

    return reqs.ok();
}

//==============================================================================
// TRMM - Triangular matrix-matrix multiply
//==============================================================================

PLATFORM_IMPL(trmm, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);  // triangular matrix
    auto b = INPUT_VARIABLE(1);  // general matrix (also output)

    double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

    bool leftSide = block.getIArguments()->size() > 0 ? (INT_ARG(0) == 0) : true;
    bool upperTriangle = block.getIArguments()->size() > 1 ? (INT_ARG(1) == 0) : true;
    bool transpose = block.getIArguments()->size() > 2 ? (INT_ARG(2) != 0) : false;
    bool unitDiag = block.getIArguments()->size() > 3 ? (INT_ARG(3) != 0) : false;

    const int m = static_cast<int>(b->sizeAt(0));
    const int n = static_cast<int>(b->sizeAt(1));
    const int lda = leftSide ? m : n;
    const int ldb = n;

    if (a->dataType() == DataType::FLOAT32) {
        cblas_strmm(CblasRowMajor,
                    leftSide ? CblasLeft : CblasRight,
                    upperTriangle ? CblasUpper : CblasLower,
                    transpose ? CblasTrans : CblasNoTrans,
                    unitDiag ? CblasUnit : CblasNonUnit,
                    m, n,
                    static_cast<float>(alpha),
                    a->bufferAsT<float>(), lda,
                    b->bufferAsT<float>(), ldb);
    } else if (a->dataType() == DataType::DOUBLE) {
        cblas_dtrmm(CblasRowMajor,
                    leftSide ? CblasLeft : CblasRight,
                    upperTriangle ? CblasUpper : CblasLower,
                    transpose ? CblasTrans : CblasNoTrans,
                    unitDiag ? CblasUnit : CblasNonUnit,
                    m, n,
                    alpha,
                    a->bufferAsT<double>(), lda,
                    b->bufferAsT<double>(), ldb);
    }

    return Status::OK;
}

PLATFORM_CHECK(trmm, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);

    Requirements reqs("TRMM ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, a, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*a), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*b), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(a->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(a->rankOf() == 2, RANK_MSG_INPUT0);
    reqs.expectTrue(b->rankOf() == 2, RANK_MSG_INPUT1);
    // a must be square
    reqs.expectTrue(a->sizeAt(0) == a->sizeAt(1), "Triangular matrix must be square");

    return reqs.ok();
}

//==============================================================================
// TRSM - Triangular solve: solve A * X = B or X * A = B
//==============================================================================

PLATFORM_IMPL(trsm, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);  // triangular matrix
    auto b = INPUT_VARIABLE(1);  // right-hand side (also output X)

    double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

    bool leftSide = block.getIArguments()->size() > 0 ? (INT_ARG(0) == 0) : true;
    bool upperTriangle = block.getIArguments()->size() > 1 ? (INT_ARG(1) == 0) : true;
    bool transpose = block.getIArguments()->size() > 2 ? (INT_ARG(2) != 0) : false;
    bool unitDiag = block.getIArguments()->size() > 3 ? (INT_ARG(3) != 0) : false;

    const int m = static_cast<int>(b->sizeAt(0));
    const int n = static_cast<int>(b->sizeAt(1));
    const int lda = leftSide ? m : n;
    const int ldb = n;

    if (a->dataType() == DataType::FLOAT32) {
        cblas_strsm(CblasRowMajor,
                    leftSide ? CblasLeft : CblasRight,
                    upperTriangle ? CblasUpper : CblasLower,
                    transpose ? CblasTrans : CblasNoTrans,
                    unitDiag ? CblasUnit : CblasNonUnit,
                    m, n,
                    static_cast<float>(alpha),
                    a->bufferAsT<float>(), lda,
                    b->bufferAsT<float>(), ldb);
    } else if (a->dataType() == DataType::DOUBLE) {
        cblas_dtrsm(CblasRowMajor,
                    leftSide ? CblasLeft : CblasRight,
                    upperTriangle ? CblasUpper : CblasLower,
                    transpose ? CblasTrans : CblasNoTrans,
                    unitDiag ? CblasUnit : CblasNonUnit,
                    m, n,
                    alpha,
                    a->bufferAsT<double>(), lda,
                    b->bufferAsT<double>(), ldb);
    }

    return Status::OK;
}

PLATFORM_CHECK(trsm, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);

    Requirements reqs("TRSM ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, a, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*a), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*b), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(a->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(a->rankOf() == 2, RANK_MSG_INPUT0);
    reqs.expectTrue(b->rankOf() == 2, RANK_MSG_INPUT1);
    reqs.expectTrue(a->sizeAt(0) == a->sizeAt(1), "Triangular matrix must be square");

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
