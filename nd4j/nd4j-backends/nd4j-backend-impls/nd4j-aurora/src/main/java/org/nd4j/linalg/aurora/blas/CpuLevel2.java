/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */


package org.nd4j.linalg.aurora.blas;


import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.blas.impl.BaseLevel2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.Nd4jAuroraOps;
import org.nd4j.nativeblas.Nd4jBlas;
import org.nd4j.nativeblas.NativeOpsHolder;


/**
 * @author Adam Gibson
 */
public class CpuLevel2 extends BaseLevel2 {
    private Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();
    Nd4jAuroraOps nativeOps = (Nd4jAuroraOps)NativeOpsHolder.getInstance().getDeviceNativeOps();

    @Override
    protected void sgemv(char order, char TransA, int M, int N, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY) {
        nativeOps.call("cblas_sgemv", CpuBlas.convertOrder('f'), CpuBlas.convertTranspose(TransA), M, N, alpha, (FloatPointer) A.data().addressPointer(),
                        lda, (FloatPointer) X.data().addressPointer(), incX, beta,
                        (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void sgbmv(char order, char TransA, int M, int N, int KL, int KU, float alpha, INDArray A, int lda,
                    INDArray X, int incX, float beta, INDArray Y, int incY) {
        nativeOps.call("cblas_sgbmv", CpuBlas.convertOrder('f'), CpuBlas.convertTranspose(TransA), M, N, KL, KU, alpha,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) X.data().addressPointer(), incX,
                        beta, (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void strmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void stbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        nativeOps.call("cblas_stbmv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N, K,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void stpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        nativeOps.call("cblas_stpmv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N,
                        (FloatPointer) Ap.data().addressPointer(), (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void strsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        nativeOps.call("cblas_strsv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void stbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        nativeOps.call("cblas_stbsv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N, K,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) X.data().addressPointer(), incX);

    }

    @Override
    protected void stpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        nativeOps.call("cblas_stpsv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N,
                        (FloatPointer) Ap.data().addressPointer(), (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void dgemv(char order, char TransA, int M, int N, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY) {
        nativeOps.call("cblas_dgemv", CpuBlas.convertOrder('f'), CpuBlas.convertTranspose(TransA), M, N, alpha, (DoublePointer) A.data().addressPointer(),
                        lda, (DoublePointer) X.data().addressPointer(), incX, beta,
                        (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dgbmv(char order, char TransA, int M, int N, int KL, int KU, double alpha, INDArray A, int lda,
                    INDArray X, int incX, double beta, INDArray Y, int incY) {
        nativeOps.call("cblas_dgbmv", CpuBlas.convertOrder('f'), CpuBlas.convertTranspose(TransA), M, N, KL, KU, alpha,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(), incX,
                        beta, (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dtrmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        nativeOps.call("cblas_dtrmv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(),
                        incX);
    }

    @Override
    protected void dtbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        nativeOps.call("cblas_dtbmv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N, K,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(),
                        incX);
    }

    @Override
    protected void dtpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        nativeOps.call("cblas_dtpmv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N,
                        (DoublePointer) Ap.data().addressPointer(), (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void dtrsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        nativeOps.call("cblas_dtrsv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(),
                        incX);
    }

    @Override
    protected void dtbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        nativeOps.call("cblas_dtbsv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N, K,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(),
                        incX);
    }

    @Override
    protected void dtpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        nativeOps.call("cblas_dtpsv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), CpuBlas.convertTranspose(TransA), CpuBlas.convertDiag(Diag), N,
                        (DoublePointer) Ap.data().addressPointer(), (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void ssymv(char order, char Uplo, int N, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY) {
        nativeOps.call("cblas_ssymv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (FloatPointer) A.data().addressPointer(), lda,
                        (FloatPointer) X.data().addressPointer(), incX, beta, (FloatPointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected void ssbmv(char order, char Uplo, int N, int K, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY) {
        nativeOps.call("cblas_ssbmv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, K, alpha, (FloatPointer) A.data().addressPointer(), lda,
                        (FloatPointer) X.data().addressPointer(), incX, beta, (FloatPointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected void sspmv(char order, char Uplo, int N, float alpha, INDArray Ap, INDArray X, int incX, float beta,
                    INDArray Y, int incY) {
        nativeOps.call("cblas_sspmv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (FloatPointer) Ap.data().addressPointer(),
                        (FloatPointer) X.data().addressPointer(), incX, beta, (FloatPointer) Y.data().addressPointer(),
                        incY);

    }

    @Override
    protected void sger(char order, int M, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A,
                    int lda) {
        nativeOps.call("cblas_sger", CpuBlas.convertOrder('f'), M, N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Y.data().addressPointer(), incY, (FloatPointer) A.data().addressPointer(), lda);
    }

    @Override
    protected void ssyr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray A, int lda) {
        nativeOps.call("cblas_ssyr", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) A.data().addressPointer(), lda);
    }

    @Override
    protected void sspr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Ap) {
        nativeOps.call("cblas_sspr", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Ap.data().addressPointer());
    }

    @Override
    protected void ssyr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A, int lda) {
        nativeOps.call("cblas_ssyr2", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Y.data().addressPointer(), incY, (FloatPointer) A.data().addressPointer(), lda);
    }

    @Override
    protected void sspr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A) {
        nativeOps.call("cblas_sspr2", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Y.data().addressPointer(), incY, (FloatPointer) A.data().addressPointer());
    }

    @Override
    protected void dsymv(char order, char Uplo, int N, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY) {
        nativeOps.call("cblas_dsymv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (DoublePointer) A.data().addressPointer(), lda,
                        (DoublePointer) X.data().addressPointer(), incX, beta,
                        (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dsbmv(char order, char Uplo, int N, int K, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY) {
        nativeOps.call("cblas_dsbmv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, K, alpha, (DoublePointer) A.data().addressPointer(), lda,
                        (DoublePointer) X.data().addressPointer(), incX, beta,
                        (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dspmv(char order, char Uplo, int N, double alpha, INDArray Ap, INDArray X, int incX, double beta,
                    INDArray Y, int incY) {
        nativeOps.call("cblas_dspmv", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (DoublePointer) Ap.data().addressPointer(),
                        (DoublePointer) X.data().addressPointer(), incX, beta,
                        (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dger(char order, int M, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A,
                    int lda) {
        nativeOps.call("cblas_dger", CpuBlas.convertOrder('f'), M, N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Y.data().addressPointer(), incY, (DoublePointer) A.data().addressPointer(),
                        lda);
    }

    @Override
    protected void dsyr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray A, int lda) {
        nativeOps.call("cblas_dsyr", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) A.data().addressPointer(), lda);
    }

    @Override
    protected void dspr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Ap) {
        nativeOps.call("cblas_dspr", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Ap.data().addressPointer());
    }

    @Override
    protected void dsyr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A, int lda) {
        nativeOps.call("cblas_dsyr2", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Y.data().addressPointer(), incY, (DoublePointer) A.data().addressPointer(),
                        lda);
    }

    @Override
    protected void dspr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A) {
        nativeOps.call("cblas_dspr2", CpuBlas.convertOrder('f'), CpuBlas.convertUplo(Uplo), N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Y.data().addressPointer(), incY, (DoublePointer) A.data().addressPointer());
    }
}
