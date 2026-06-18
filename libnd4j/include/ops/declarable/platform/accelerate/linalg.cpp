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
// Apple Accelerate framework - Linear Algebra operations via LAPACK
// Provides LU, Cholesky, QR, SVD decompositions, and linear system solvers
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE

// LAPACK function declarations
extern "C" {
    // LU decomposition
    int sgetrf_(__CLPK_integer* m, __CLPK_integer* n, float* a, __CLPK_integer* lda,
                __CLPK_integer* ipiv, __CLPK_integer* info);
    int dgetrf_(__CLPK_integer* m, __CLPK_integer* n, double* a, __CLPK_integer* lda,
                __CLPK_integer* ipiv, __CLPK_integer* info);

    // Matrix inverse (after LU)
    int sgetri_(__CLPK_integer* n, float* a, __CLPK_integer* lda, __CLPK_integer* ipiv,
                float* work, __CLPK_integer* lwork, __CLPK_integer* info);
    int dgetri_(__CLPK_integer* n, double* a, __CLPK_integer* lda, __CLPK_integer* ipiv,
                double* work, __CLPK_integer* lwork, __CLPK_integer* info);

    // Cholesky decomposition
    int spotrf_(char* uplo, __CLPK_integer* n, float* a, __CLPK_integer* lda, __CLPK_integer* info);
    int dpotrf_(char* uplo, __CLPK_integer* n, double* a, __CLPK_integer* lda, __CLPK_integer* info);

    // QR decomposition
    int sgeqrf_(__CLPK_integer* m, __CLPK_integer* n, float* a, __CLPK_integer* lda,
                float* tau, float* work, __CLPK_integer* lwork, __CLPK_integer* info);
    int dgeqrf_(__CLPK_integer* m, __CLPK_integer* n, double* a, __CLPK_integer* lda,
                double* tau, double* work, __CLPK_integer* lwork, __CLPK_integer* info);

    // Generate Q from QR
    int sorgqr_(__CLPK_integer* m, __CLPK_integer* n, __CLPK_integer* k, float* a,
                __CLPK_integer* lda, float* tau, float* work, __CLPK_integer* lwork, __CLPK_integer* info);
    int dorgqr_(__CLPK_integer* m, __CLPK_integer* n, __CLPK_integer* k, double* a,
                __CLPK_integer* lda, double* tau, double* work, __CLPK_integer* lwork, __CLPK_integer* info);

    // SVD
    int sgesvd_(char* jobu, char* jobvt, __CLPK_integer* m, __CLPK_integer* n, float* a,
                __CLPK_integer* lda, float* s, float* u, __CLPK_integer* ldu, float* vt,
                __CLPK_integer* ldvt, float* work, __CLPK_integer* lwork, __CLPK_integer* info);
    int dgesvd_(char* jobu, char* jobvt, __CLPK_integer* m, __CLPK_integer* n, double* a,
                __CLPK_integer* lda, double* s, double* u, __CLPK_integer* ldu, double* vt,
                __CLPK_integer* ldvt, double* work, __CLPK_integer* lwork, __CLPK_integer* info);

    // Linear solve (general)
    int sgesv_(__CLPK_integer* n, __CLPK_integer* nrhs, float* a, __CLPK_integer* lda,
               __CLPK_integer* ipiv, float* b, __CLPK_integer* ldb, __CLPK_integer* info);
    int dgesv_(__CLPK_integer* n, __CLPK_integer* nrhs, double* a, __CLPK_integer* lda,
               __CLPK_integer* ipiv, double* b, __CLPK_integer* ldb, __CLPK_integer* info);

    // Eigenvalues (symmetric)
    int ssyev_(char* jobz, char* uplo, __CLPK_integer* n, float* a, __CLPK_integer* lda,
               float* w, float* work, __CLPK_integer* lwork, __CLPK_integer* info);
    int dsyev_(char* jobz, char* uplo, __CLPK_integer* n, double* a, __CLPK_integer* lda,
               double* w, double* work, __CLPK_integer* lwork, __CLPK_integer* info);
}

namespace sd {
namespace ops {
namespace platforms {

//==============================================================================
// LU - LU Decomposition: A = P * L * U
//==============================================================================

PLATFORM_IMPL(lu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto outputLU = OUTPUT_VARIABLE(0);
    auto outputP = OUTPUT_NULLIFIED(1);  // Permutation indices

    const __CLPK_integer m = static_cast<__CLPK_integer>(input->sizeAt(-2));
    const __CLPK_integer n = static_cast<__CLPK_integer>(input->sizeAt(-1));
    const __CLPK_integer lda = n;
    const __CLPK_integer minMN = std::min(m, n);

    // Copy input to output (LAPACK overwrites)
    outputLU->assign(input);

    std::vector<__CLPK_integer> ipiv(minMN);
    __CLPK_integer info = 0;

    if (input->dataType() == DataType::FLOAT32) {
        sgetrf_(&m, &n, outputLU->bufferAsT<float>(), &lda, ipiv.data(), &info);
    } else if (input->dataType() == DataType::DOUBLE) {
        dgetrf_(&m, &n, outputLU->bufferAsT<double>(), &lda, ipiv.data(), &info);
    }

    if (info != 0) {
        return Status::KERNEL_FAILURE;
    }

    // Convert to permutation output if requested
    if (outputP != nullptr) {
        for (__CLPK_integer i = 0; i < minMN; i++) {
            outputP->p(i, static_cast<LongType>(ipiv[i] - 1));  // Convert from 1-indexed
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(lu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("LU ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 2, RANK_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// CHOLESKY - Cholesky Decomposition: A = L * L^T (for positive definite)
//==============================================================================

PLATFORM_IMPL(cholesky, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const __CLPK_integer n = static_cast<__CLPK_integer>(input->sizeAt(0));
    const __CLPK_integer lda = n;
    char uplo = 'L';  // Lower triangular
    __CLPK_integer info = 0;

    // Copy input to output
    output->assign(input);

    if (input->dataType() == DataType::FLOAT32) {
        spotrf_(&uplo, &n, output->bufferAsT<float>(), &lda, &info);

        // Zero out upper triangle
        float* buf = output->bufferAsT<float>();
        for (__CLPK_integer i = 0; i < n; i++) {
            for (__CLPK_integer j = i + 1; j < n; j++) {
                buf[i * n + j] = 0.0f;
            }
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        dpotrf_(&uplo, &n, output->bufferAsT<double>(), &lda, &info);

        double* buf = output->bufferAsT<double>();
        for (__CLPK_integer i = 0; i < n; i++) {
            for (__CLPK_integer j = i + 1; j < n; j++) {
                buf[i * n + j] = 0.0;
            }
        }
    }

    if (info != 0) {
        return Status::KERNEL_FAILURE;
    }

    return Status::OK;
}

PLATFORM_CHECK(cholesky, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("CHOLESKY ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 2, RANK_MSG_INPUT0);
    reqs.expectTrue(input->sizeAt(0) == input->sizeAt(1), "Input must be square matrix");

    return reqs.ok();
}

//==============================================================================
// QR - QR Decomposition: A = Q * R
//==============================================================================

PLATFORM_IMPL(qr, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto outputQ = OUTPUT_VARIABLE(0);
    auto outputR = OUTPUT_VARIABLE(1);

    const __CLPK_integer m = static_cast<__CLPK_integer>(input->sizeAt(0));
    const __CLPK_integer n = static_cast<__CLPK_integer>(input->sizeAt(1));
    const __CLPK_integer lda = n;
    const __CLPK_integer k = std::min(m, n);
    __CLPK_integer info = 0;

    // Copy input to working array
    std::vector<float> workArrayF;
    std::vector<double> workArrayD;

    if (input->dataType() == DataType::FLOAT32) {
        workArrayF.resize(m * n);
        std::memcpy(workArrayF.data(), input->buffer(), m * n * sizeof(float));

        std::vector<float> tau(k);

        // Query optimal workspace size
        __CLPK_integer lwork = -1;
        float workQuery;
        sgeqrf_(&m, &n, workArrayF.data(), &lda, tau.data(), &workQuery, &lwork, &info);

        lwork = static_cast<__CLPK_integer>(workQuery);
        std::vector<float> work(lwork);

        // Compute QR
        sgeqrf_(&m, &n, workArrayF.data(), &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0) return Status::KERNEL_FAILURE;

        // Extract R (upper triangular)
        float* rBuf = outputR->bufferAsT<float>();
        for (__CLPK_integer i = 0; i < k; i++) {
            for (__CLPK_integer j = 0; j < n; j++) {
                if (j >= i) {
                    rBuf[i * n + j] = workArrayF[i * n + j];
                } else {
                    rBuf[i * n + j] = 0.0f;
                }
            }
        }

        // Generate Q
        __CLPK_integer mQ = m;
        __CLPK_integer nQ = k;
        sorgqr_(&mQ, &nQ, &k, workArrayF.data(), &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0) return Status::KERNEL_FAILURE;

        // Copy Q
        float* qBuf = outputQ->bufferAsT<float>();
        for (__CLPK_integer i = 0; i < m; i++) {
            for (__CLPK_integer j = 0; j < k; j++) {
                qBuf[i * k + j] = workArrayF[i * n + j];
            }
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        workArrayD.resize(m * n);
        std::memcpy(workArrayD.data(), input->buffer(), m * n * sizeof(double));

        std::vector<double> tau(k);

        __CLPK_integer lwork = -1;
        double workQuery;
        dgeqrf_(&m, &n, workArrayD.data(), &lda, tau.data(), &workQuery, &lwork, &info);

        lwork = static_cast<__CLPK_integer>(workQuery);
        std::vector<double> work(lwork);

        dgeqrf_(&m, &n, workArrayD.data(), &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0) return Status::KERNEL_FAILURE;

        double* rBuf = outputR->bufferAsT<double>();
        for (__CLPK_integer i = 0; i < k; i++) {
            for (__CLPK_integer j = 0; j < n; j++) {
                if (j >= i) {
                    rBuf[i * n + j] = workArrayD[i * n + j];
                } else {
                    rBuf[i * n + j] = 0.0;
                }
            }
        }

        __CLPK_integer mQ = m;
        __CLPK_integer nQ = k;
        dorgqr_(&mQ, &nQ, &k, workArrayD.data(), &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0) return Status::KERNEL_FAILURE;

        double* qBuf = outputQ->bufferAsT<double>();
        for (__CLPK_integer i = 0; i < m; i++) {
            for (__CLPK_integer j = 0; j < k; j++) {
                qBuf[i * k + j] = workArrayD[i * n + j];
            }
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(qr, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements reqs("QR ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 2, RANK_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SVD - Singular Value Decomposition: A = U * S * V^T
//==============================================================================

PLATFORM_IMPL(svd, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    // Check what outputs are requested
    bool computeUV = block.getBArguments()->size() > 0 ? B_ARG(0) : true;
    bool fullMatrices = block.getBArguments()->size() > 1 ? B_ARG(1) : false;

    auto outputS = OUTPUT_VARIABLE(0);  // Singular values
    auto outputU = computeUV ? OUTPUT_VARIABLE(1) : nullptr;
    auto outputV = computeUV ? OUTPUT_VARIABLE(2) : nullptr;

    const __CLPK_integer m = static_cast<__CLPK_integer>(input->sizeAt(0));
    const __CLPK_integer n = static_cast<__CLPK_integer>(input->sizeAt(1));
    const __CLPK_integer lda = n;
    const __CLPK_integer minMN = std::min(m, n);

    char jobu = computeUV ? (fullMatrices ? 'A' : 'S') : 'N';
    char jobvt = computeUV ? (fullMatrices ? 'A' : 'S') : 'N';

    __CLPK_integer ldu = m;
    __CLPK_integer ldvt = fullMatrices ? n : minMN;
    __CLPK_integer info = 0;

    if (input->dataType() == DataType::FLOAT32) {
        // Copy input (LAPACK destroys it)
        std::vector<float> aCopy(m * n);
        std::memcpy(aCopy.data(), input->buffer(), m * n * sizeof(float));

        std::vector<float> u(computeUV ? m * (fullMatrices ? m : minMN) : 1);
        std::vector<float> vt(computeUV ? (fullMatrices ? n : minMN) * n : 1);

        // Query workspace
        __CLPK_integer lwork = -1;
        float workQuery;
        sgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &lda,
                outputS->bufferAsT<float>(), u.data(), &ldu, vt.data(), &ldvt,
                &workQuery, &lwork, &info);

        lwork = static_cast<__CLPK_integer>(workQuery);
        std::vector<float> work(lwork);

        sgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &lda,
                outputS->bufferAsT<float>(), u.data(), &ldu, vt.data(), &ldvt,
                work.data(), &lwork, &info);

        if (info != 0) return Status::KERNEL_FAILURE;

        if (computeUV) {
            std::memcpy(outputU->buffer(), u.data(), outputU->lengthOf() * sizeof(float));
            // Transpose V^T to get V
            float* vBuf = outputV->bufferAsT<float>();
            __CLPK_integer vRows = fullMatrices ? n : minMN;
            for (__CLPK_integer i = 0; i < vRows; i++) {
                for (__CLPK_integer j = 0; j < n; j++) {
                    vBuf[j * vRows + i] = vt[i * n + j];
                }
            }
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        std::vector<double> aCopy(m * n);
        std::memcpy(aCopy.data(), input->buffer(), m * n * sizeof(double));

        std::vector<double> u(computeUV ? m * (fullMatrices ? m : minMN) : 1);
        std::vector<double> vt(computeUV ? (fullMatrices ? n : minMN) * n : 1);

        __CLPK_integer lwork = -1;
        double workQuery;
        dgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &lda,
                outputS->bufferAsT<double>(), u.data(), &ldu, vt.data(), &ldvt,
                &workQuery, &lwork, &info);

        lwork = static_cast<__CLPK_integer>(workQuery);
        std::vector<double> work(lwork);

        dgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &lda,
                outputS->bufferAsT<double>(), u.data(), &ldu, vt.data(), &ldvt,
                work.data(), &lwork, &info);

        if (info != 0) return Status::KERNEL_FAILURE;

        if (computeUV) {
            std::memcpy(outputU->buffer(), u.data(), outputU->lengthOf() * sizeof(double));
            double* vBuf = outputV->bufferAsT<double>();
            __CLPK_integer vRows = fullMatrices ? n : minMN;
            for (__CLPK_integer i = 0; i < vRows; i++) {
                for (__CLPK_integer j = 0; j < n; j++) {
                    vBuf[j * vRows + i] = vt[i * n + j];
                }
            }
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(svd, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements reqs("SVD ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 2, RANK_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SOLVE - Linear System Solver: solve A * X = B for X
//==============================================================================

PLATFORM_IMPL(solve, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);  // Coefficient matrix A
    auto b = INPUT_VARIABLE(1);  // Right-hand side B
    auto x = OUTPUT_VARIABLE(0); // Solution X

    const __CLPK_integer n = static_cast<__CLPK_integer>(a->sizeAt(0));
    const __CLPK_integer nrhs = b->rankOf() == 1 ? 1 : static_cast<__CLPK_integer>(b->sizeAt(1));
    const __CLPK_integer lda = n;
    const __CLPK_integer ldb = b->rankOf() == 1 ? n : static_cast<__CLPK_integer>(b->sizeAt(1));
    __CLPK_integer info = 0;

    std::vector<__CLPK_integer> ipiv(n);

    // Copy inputs (LAPACK destroys them)
    if (a->dataType() == DataType::FLOAT32) {
        std::vector<float> aCopy(n * n);
        std::memcpy(aCopy.data(), a->buffer(), n * n * sizeof(float));

        x->assign(b);

        sgesv_(&n, &nrhs, aCopy.data(), &lda, ipiv.data(),
               x->bufferAsT<float>(), &ldb, &info);
    } else if (a->dataType() == DataType::DOUBLE) {
        std::vector<double> aCopy(n * n);
        std::memcpy(aCopy.data(), a->buffer(), n * n * sizeof(double));

        x->assign(b);

        dgesv_(&n, &nrhs, aCopy.data(), &lda, ipiv.data(),
               x->bufferAsT<double>(), &ldb, &info);
    }

    if (info != 0) {
        return Status::KERNEL_FAILURE;
    }

    return Status::OK;
}

PLATFORM_CHECK(solve, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);

    Requirements reqs("SOLVE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, a, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*a), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*b), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(a->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(a->rankOf() == 2, RANK_MSG_INPUT0);
    reqs.expectTrue(a->sizeAt(0) == a->sizeAt(1), "Coefficient matrix must be square");

    return reqs.ok();
}

//==============================================================================
// MATRIX_INVERSE - Matrix Inverse via LU decomposition
//==============================================================================

PLATFORM_IMPL(matrix_inverse, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const __CLPK_integer n = static_cast<__CLPK_integer>(input->sizeAt(0));
    const __CLPK_integer lda = n;
    __CLPK_integer info = 0;

    // Copy input to output
    output->assign(input);

    std::vector<__CLPK_integer> ipiv(n);

    if (input->dataType() == DataType::FLOAT32) {
        // LU decomposition
        sgetrf_(&n, &n, output->bufferAsT<float>(), &lda, ipiv.data(), &info);
        if (info != 0) return Status::KERNEL_FAILURE;

        // Query workspace
        __CLPK_integer lwork = -1;
        float workQuery;
        sgetri_(&n, output->bufferAsT<float>(), &lda, ipiv.data(), &workQuery, &lwork, &info);

        lwork = static_cast<__CLPK_integer>(workQuery);
        std::vector<float> work(lwork);

        // Compute inverse
        sgetri_(&n, output->bufferAsT<float>(), &lda, ipiv.data(), work.data(), &lwork, &info);
    } else if (input->dataType() == DataType::DOUBLE) {
        dgetrf_(&n, &n, output->bufferAsT<double>(), &lda, ipiv.data(), &info);
        if (info != 0) return Status::KERNEL_FAILURE;

        __CLPK_integer lwork = -1;
        double workQuery;
        dgetri_(&n, output->bufferAsT<double>(), &lda, ipiv.data(), &workQuery, &lwork, &info);

        lwork = static_cast<__CLPK_integer>(workQuery);
        std::vector<double> work(lwork);

        dgetri_(&n, output->bufferAsT<double>(), &lda, ipiv.data(), work.data(), &lwork, &info);
    }

    if (info != 0) {
        return Status::KERNEL_FAILURE;
    }

    return Status::OK;
}

PLATFORM_CHECK(matrix_inverse, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("MATRIX_INVERSE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 2, RANK_MSG_INPUT0);
    reqs.expectTrue(input->sizeAt(0) == input->sizeAt(1), "Input must be square matrix");

    return reqs.ok();
}

//==============================================================================
// EIGVALSH - Eigenvalues of symmetric matrix
//==============================================================================

PLATFORM_IMPL(eigvalsh, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto outputW = OUTPUT_VARIABLE(0);  // Eigenvalues
    auto outputV = block.width() > 1 ? OUTPUT_VARIABLE(1) : nullptr;  // Eigenvectors (optional)

    const __CLPK_integer n = static_cast<__CLPK_integer>(input->sizeAt(0));
    const __CLPK_integer lda = n;
    char jobz = outputV != nullptr ? 'V' : 'N';
    char uplo = 'U';
    __CLPK_integer info = 0;

    // Copy input (LAPACK destroys it, but if we want eigenvectors, we need it)
    std::vector<float> aCopyF;
    std::vector<double> aCopyD;

    if (input->dataType() == DataType::FLOAT32) {
        aCopyF.resize(n * n);
        std::memcpy(aCopyF.data(), input->buffer(), n * n * sizeof(float));

        __CLPK_integer lwork = -1;
        float workQuery;
        ssyev_(&jobz, &uplo, &n, aCopyF.data(), &lda, outputW->bufferAsT<float>(),
               &workQuery, &lwork, &info);

        lwork = static_cast<__CLPK_integer>(workQuery);
        std::vector<float> work(lwork);

        ssyev_(&jobz, &uplo, &n, aCopyF.data(), &lda, outputW->bufferAsT<float>(),
               work.data(), &lwork, &info);

        if (outputV != nullptr) {
            std::memcpy(outputV->buffer(), aCopyF.data(), n * n * sizeof(float));
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        aCopyD.resize(n * n);
        std::memcpy(aCopyD.data(), input->buffer(), n * n * sizeof(double));

        __CLPK_integer lwork = -1;
        double workQuery;
        dsyev_(&jobz, &uplo, &n, aCopyD.data(), &lda, outputW->bufferAsT<double>(),
               &workQuery, &lwork, &info);

        lwork = static_cast<__CLPK_integer>(workQuery);
        std::vector<double> work(lwork);

        dsyev_(&jobz, &uplo, &n, aCopyD.data(), &lda, outputW->bufferAsT<double>(),
               work.data(), &lwork, &info);

        if (outputV != nullptr) {
            std::memcpy(outputV->buffer(), aCopyD.data(), n * n * sizeof(double));
        }
    }

    if (info != 0) {
        return Status::KERNEL_FAILURE;
    }

    return Status::OK;
}

PLATFORM_CHECK(eigvalsh, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements reqs("EIGVALSH ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 2, RANK_MSG_INPUT0);
    reqs.expectTrue(input->sizeAt(0) == input->sizeAt(1), "Input must be square matrix");

    return reqs.ok();
}

//==============================================================================
// MATRIX_DETERMINANT - Determinant via LU decomposition
//==============================================================================

PLATFORM_IMPL(matrix_determinant, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const __CLPK_integer n = static_cast<__CLPK_integer>(input->sizeAt(0));
    const __CLPK_integer lda = n;
    __CLPK_integer info = 0;

    std::vector<__CLPK_integer> ipiv(n);

    if (input->dataType() == DataType::FLOAT32) {
        std::vector<float> luCopy(n * n);
        std::memcpy(luCopy.data(), input->buffer(), n * n * sizeof(float));

        sgetrf_(&n, &n, luCopy.data(), &lda, ipiv.data(), &info);

        if (info != 0) {
            output->p(0, 0.0f);
            return Status::OK;
        }

        // Determinant is product of diagonal elements * sign from pivoting
        float det = 1.0f;
        int sign = 1;
        for (__CLPK_integer i = 0; i < n; i++) {
            det *= luCopy[i * n + i];
            if (ipiv[i] != i + 1) sign = -sign;
        }
        output->p(0, det * sign);
    } else if (input->dataType() == DataType::DOUBLE) {
        std::vector<double> luCopy(n * n);
        std::memcpy(luCopy.data(), input->buffer(), n * n * sizeof(double));

        dgetrf_(&n, &n, luCopy.data(), &lda, ipiv.data(), &info);

        if (info != 0) {
            output->p(0, 0.0);
            return Status::OK;
        }

        double det = 1.0;
        int sign = 1;
        for (__CLPK_integer i = 0; i < n; i++) {
            det *= luCopy[i * n + i];
            if (ipiv[i] != i + 1) sign = -sign;
        }
        output->p(0, det * sign);
    }

    return Status::OK;
}

PLATFORM_CHECK(matrix_determinant, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("MATRIX_DETERMINANT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 2, RANK_MSG_INPUT0);
    reqs.expectTrue(input->sizeAt(0) == input->sizeAt(1), "Input must be square matrix");
    reqs.expectTrue(output->isScalar(), "Output must be scalar");

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
