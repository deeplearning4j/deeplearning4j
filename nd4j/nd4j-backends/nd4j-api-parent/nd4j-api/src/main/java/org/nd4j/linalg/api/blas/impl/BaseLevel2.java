/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.Level2;
import org.nd4j.linalg.api.blas.params.GemvParameters;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;

/**
 * Base class for level 2 functions, abstract headers pulled from:
 * http://www.netlib.org/blas/blast-forum/cblas.h
 *
 * @author Adam Gibson
 */
public abstract class BaseLevel2 extends BaseLevel implements Level2 {
    /**
     * gemv computes a matrix-vector product using a general matrix and performs one of the following matrix-vector operations:
     * y := alpha*a*x + beta*y  for trans = 'N'or'n';
     * y := alpha*a'*x + beta*y  for trans = 'T'or't';
     * y := alpha*conjg(a')*x + beta*y  for trans = 'C'or'c'.
     * Here a is an m-by-n band matrix, x and y are vectors, alpha and beta are scalars.
     *
     * @param order
     * @param transA
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void gemv(char order, char transA, double alpha, INDArray A, INDArray X, double beta, INDArray Y) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X, Y);

        if (A.isSparse() && !X.isSparse()) {
            Nd4j.getSparseBlasWrapper().level2().gemv(order, transA, alpha, A, X, beta, Y);
            return;
        }

        GemvParameters parameters = new GemvParameters(A, X, Y);
        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, parameters.getA(), parameters.getX(),
                            parameters.getY());
            dgemv(order, parameters.getAOrdering(), parameters.getM(), parameters.getN(), alpha, parameters.getA(),
                            parameters.getLda(), parameters.getX(), parameters.getIncx(), beta, parameters.getY(),
                            parameters.getIncy());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, parameters.getA(), parameters.getX(),
                            parameters.getY());
            sgemv(order, parameters.getAOrdering(), parameters.getM(), parameters.getN(), (float) alpha,
                            parameters.getA(), parameters.getLda(), parameters.getX(), parameters.getIncx(),
                            (float) beta, parameters.getY(), parameters.getIncy());
        }

        OpExecutionerUtil.checkForAny(Y);
    }

    /**
     * gbmv computes a matrix-vector product using a general band matrix and performs one of the following matrix-vector operations:
     * y := alpha*a*x + beta*y  for trans = 'N'or'n';
     * y := alpha*a'*x + beta*y  for trans = 'T'or't';
     * y := alpha*conjg(a')*x + beta*y  for trans = 'C'or'c'.
     * Here a is an m-by-n band matrix with ku superdiagonals and kl subdiagonals, x and y are vectors, alpha and beta are scalars.
     *
     * @param order
     * @param TransA
     * @param KL
     * @param KU
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void gbmv(char order, char TransA, int KL, int KU, double alpha, INDArray A, INDArray X, double beta,
                    INDArray Y) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X, Y);

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X, Y);
            dgbmv(order, TransA, (int) A.rows(), (int) A.columns(), KL, KU, alpha, A, (int) A.size(0), X, X.majorStride(), beta, Y,
                            Y.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X, Y);
            sgbmv(order, TransA, (int) A.rows(), (int) A.columns(), KL, KU, (float) alpha, A, (int) A.size(0), X, X.majorStride(), (float) beta, Y, Y.majorStride());
        }

        OpExecutionerUtil.checkForAny(Y);
    }

    /**
     * performs a rank-1 update of a general m-by-n matrix a:
     * a := alpha*x*y' + a.
     *
     * @param order
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    @Override
    public void ger(char order, double alpha, INDArray X, INDArray Y, INDArray A) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X, Y);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X, Y);
            dger(order, (int) A.rows(), (int) A.columns(), alpha, X, X.majorStride(), Y, Y.majorStride(), A, (int) A.size(0));
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X, Y);
            sger(order, (int) A.rows(), (int) A.columns(), (float) alpha, X, X.majorStride(), Y, Y.majorStride(), A, (int) A.size(0));
        }

        OpExecutionerUtil.checkForAny(A);
    }

    /**
     * sbmv computes a matrix-vector product using a symmetric band matrix:
     * y := alpha*a*x + beta*y.
     * Here a is an n-by-n symmetric band matrix with k superdiagonals, x and y are n-element vectors, alpha and beta are scalars.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void sbmv(char order, char Uplo, double alpha, INDArray A, INDArray X, double beta, INDArray Y) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X, Y);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X, Y);
            dsbmv(order, Uplo, (int) X.length(), (int) A.columns(), alpha, A, (int) A.size(0), X, X.majorStride(), beta, Y,
                    (int) Y.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X, Y);
            ssbmv(order, Uplo, (int) X.length(), (int) A.columns(), (float) alpha, A, (int) A.size(0), X, X.majorStride(), (float) beta,
                            Y, Y.majorStride());
        }

        OpExecutionerUtil.checkForAny(Y);
    }

    /**
     * @param order
     * @param Uplo
     * @param alpha
     * @param Ap
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void spmv(char order, char Uplo, double alpha, INDArray Ap, INDArray X, double beta, INDArray Y) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, Ap, X, Y);

        // FIXME: int cast

        if (Ap.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, X, Y);
            dspmv(order, Uplo, (int) X.length(), alpha, Ap, X, Ap.majorStride(), beta, Y, Y.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, X, Y);
            sspmv(order, Uplo, (int) X.length(), (float) alpha, Ap, X, Ap.majorStride(), (float) beta, Y, Y.majorStride());
        }

        OpExecutionerUtil.checkForAny(Y);
    }

    /**
     * spr performs a rank-1 update of an n-by-n packed symmetric matrix a:
     * a := alpha*x*x' + a.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Ap
     */
    @Override
    public void spr(char order, char Uplo, double alpha, INDArray X, INDArray Ap) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, Ap, X);


        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, X);
            dspr(order, Uplo, (int) X.length(), alpha, X, X.majorStride(), Ap);
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, X);
            sspr(order, Uplo, (int) X.length(), (float) alpha, X, X.majorStride(), Ap);
        }

        OpExecutionerUtil.checkForAny(Ap);
    }

    /**
     * ?spr2 performs a rank-2 update of an n-by-n packed symmetric matrix a:
     * a := alpha*x*y' + alpha*y*x' + a.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    @Override
    public void spr2(char order, char Uplo, double alpha, INDArray X, INDArray Y, INDArray A) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X, Y);

        // FIXME int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X, Y);
            dspr2(order, Uplo, (int) X.length(), alpha, X, X.majorStride(), Y, Y.majorStride(), A);
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X, Y);
            sspr2(order, Uplo, (int) X.length(), (float) alpha, X, X.majorStride(), Y, Y.majorStride(), A);
        }

        OpExecutionerUtil.checkForAny(A);
    }

    /**
     * symv computes a matrix-vector product for a symmetric matrix:
     * y := alpha*a*x + beta*y.
     * Here a is an n-by-n symmetric matrix; x and y are n-element vectors, alpha and beta are scalars.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void symv(char order, char Uplo, double alpha, INDArray A, INDArray X, double beta, INDArray Y) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X, Y);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X, Y);
            dsymv(order, Uplo, (int) X.length(), alpha, A, (int) A.size(0), X, X.majorStride(), beta, Y, Y.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X, Y);
            ssymv(order, Uplo, (int) X.length(), (float) alpha, A, (int) A.size(0), X, X.majorStride(), (float) beta, Y,
                            Y.majorStride());
        }

        OpExecutionerUtil.checkForAny(Y);
    }

    /**
     * syr performs a rank-1 update of an n-by-n symmetric matrix a:
     * a := alpha*x*x' + a.
     *
     * @param order
     * @param Uplo
     * @param N
     * @param alpha
     * @param X
     * @param A
     */
    @Override
    public void syr(char order, char Uplo, int N, double alpha, INDArray X, INDArray A) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X);
            dsyr(order, Uplo, (int) X.length(), alpha, X, X.majorStride(), A, (int) A.size(0));
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X);
            ssyr(order, Uplo, (int) X.length(), (float) alpha, X, X.majorStride(), A, (int) A.size(0));
        }

        OpExecutionerUtil.checkForAny(A);
    }

    /**
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    @Override
    public void syr2(char order, char Uplo, double alpha, INDArray X, INDArray Y, INDArray A) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X, Y);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X, Y);
            dsyr2(order, Uplo, (int) X.length(), alpha, X, X.majorStride(), Y, Y.majorStride(), A, (int) A.size(0));
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X, Y);
            ssyr2(order, Uplo, (int) X.length(), (float) alpha, X, X.majorStride(), Y, Y.majorStride(), A, (int) A.size(0));
        }

        OpExecutionerUtil.checkForAny(A);
    }

    /**
     * syr2 performs a rank-2 update of an n-by-n symmetric matrix a:
     * a := alpha*x*y' + alpha*y*x' + a.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    @Override
    public void tbmv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X);
            dtbmv(order, Uplo, TransA, Diag, (int) X.length(), (int) A.columns(), A, (int) A.size(0), X, X.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X);
            stbmv(order, Uplo, TransA, Diag, (int) X.length(), (int) A.columns(), A, (int) A.size(0), X, X.majorStride());
        }
    }

    /**
     * ?tbsv solves a system of linear equations whose coefficients are in a triangular band matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    @Override
    public void tbsv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X);
            dtbsv(order, Uplo, TransA, Diag, (int) X.length(), (int) A.columns(), A, (int) A.size(0), X, X.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X);
            stbsv(order, Uplo, TransA, Diag, (int) X.length(), (int) A.columns(), A, (int) A.size(0), X, X.majorStride());
        }

    }

    /**
     * tpmv computes a matrix-vector product using a triangular packed matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param Ap
     * @param X
     */
    @Override
    public void tpmv(char order, char Uplo, char TransA, char Diag, INDArray Ap, INDArray X) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, Ap, X);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, X);
            dtpmv(order, Uplo, TransA, Diag, (int) Ap.length(), Ap, X, X.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, X);
            stpmv(order, Uplo, TransA, Diag, (int) Ap.length(), Ap, X, X.majorStride());
        }

        OpExecutionerUtil.checkForAny(X);
    }

    /**
     * tpsv solves a system of linear equations whose coefficients are in a triangular packed matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param Ap
     * @param X
     */
    @Override
    public void tpsv(char order, char Uplo, char TransA, char Diag, INDArray Ap, INDArray X) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, Ap, X);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, X, Ap);
            dtpsv(order, Uplo, TransA, Diag, (int) X.length(), Ap, X, X.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, Ap, X);
            stpsv(order, Uplo, TransA, Diag, (int) X.length(), Ap, X, X.majorStride());
        }

        OpExecutionerUtil.checkForAny(X);
    }

    /**
     * trmv computes a matrix-vector product using a triangular matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    @Override
    public void trmv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X);

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X);
            dtrmv(order, Uplo, TransA, Diag, (int) X.length(), A, (int) A.size(0), X, X.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X);
            strmv(order, Uplo, TransA, Diag, (int) X.length(), A, (int) A.size(0), X, X.majorStride());
        }

        OpExecutionerUtil.checkForAny(X);
    }

    /**
     * trsv solves a system of linear equations whose coefficients are in a triangular matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    @Override
    public void trsv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, X);

        // FIXME: int cast

        if (X.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, X);
            dtrsv(order, Uplo, TransA, Diag, (int) A.length(), A, (int) A.size(0), X, X.majorStride());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, X);
            strsv(order, Uplo, TransA, Diag, (int) A.length(), A, (int) A.size(0), X, X.majorStride());
        }

        OpExecutionerUtil.checkForAny(X);
    }

    /*
     * ===========================================================================
     * Prototypes for level 2 BLAS
     * ===========================================================================
     */

    /* 
     * Routines with standard 4 prefixes (S, D, C, Z)
     */
    protected abstract void sgemv(char order, char TransA, int M, int N, float alpha, INDArray A, int lda, INDArray X,
                    int incX, float beta, INDArray Y, int incY);

    protected abstract void sgbmv(char order, char TransA, int M, int N, int KL, int KU, float alpha, INDArray A,
                    int lda, INDArray X, int incX, float beta, INDArray Y, int incY);

    protected abstract void strmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX);

    protected abstract void stbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda,
                    INDArray X, int incX);

    protected abstract void stpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X,
                    int incX);

    protected abstract void strsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX);

    protected abstract void stbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda,
                    INDArray X, int incX);

    protected abstract void stpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X,
                    int incX);

    protected abstract void dgemv(char order, char TransA, int M, int N, double alpha, INDArray A, int lda, INDArray X,
                    int incX, double beta, INDArray Y, int incY);

    protected abstract void dgbmv(char order, char TransA, int M, int N, int KL, int KU, double alpha, INDArray A,
                    int lda, INDArray X, int incX, double beta, INDArray Y, int incY);

    protected abstract void dtrmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX);

    protected abstract void dtbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda,
                    INDArray X, int incX);

    protected abstract void dtpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X,
                    int incX);

    protected abstract void dtrsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX);

    protected abstract void dtbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda,
                    INDArray X, int incX);

    protected abstract void dtpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X,
                    int incX);

    /* 
     * Routines with S and D prefixes only
     */
    protected abstract void ssymv(char order, char Uplo, int N, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY);

    protected abstract void ssbmv(char order, char Uplo, int N, int K, float alpha, INDArray A, int lda, INDArray X,
                    int incX, float beta, INDArray Y, int incY);

    protected abstract void sspmv(char order, char Uplo, int N, float alpha, INDArray Ap, INDArray X, int incX,
                    float beta, INDArray Y, int incY);

    protected abstract void sger(char order, int M, int N, float alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A, int lda);

    protected abstract void ssyr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray A, int lda);

    protected abstract void sspr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Ap);

    protected abstract void ssyr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A, int lda);

    protected abstract void sspr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A);

    protected abstract void dsymv(char order, char Uplo, int N, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY);

    protected abstract void dsbmv(char order, char Uplo, int N, int K, double alpha, INDArray A, int lda, INDArray X,
                    int incX, double beta, INDArray Y, int incY);

    protected abstract void dspmv(char order, char Uplo, int N, double alpha, INDArray Ap, INDArray X, int incX,
                    double beta, INDArray Y, int incY);

    protected abstract void dger(char order, int M, int N, double alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A, int lda);

    protected abstract void dsyr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray A, int lda);

    protected abstract void dspr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Ap);

    protected abstract void dsyr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y,
                    int incY, INDArray A, int lda);

    protected abstract void dspr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y,
                    int incY, INDArray A);
}


