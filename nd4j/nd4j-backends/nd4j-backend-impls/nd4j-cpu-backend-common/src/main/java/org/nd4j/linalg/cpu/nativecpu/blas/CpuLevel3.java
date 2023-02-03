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

package org.nd4j.linalg.cpu.nativecpu.blas;



import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.blas.impl.BaseLevel3;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.impl.AggregateGEMM;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.cpu.nativecpu.blas.CpuBlas.*;


/**
 *
 * A jblas delgation for level 3 routines
 *
 * @author Adam Gibson
 */
public class CpuLevel3 extends BaseLevel3 {

    @Override
    protected void hgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda,
                    INDArray B, int ldb, float beta, INDArray C, int ldc) {

        //if (true) {
            var fA = A.castTo(DataType.FLOAT);
            var fB = B.castTo(DataType.FLOAT);
            var fC = C.castTo(DataType.FLOAT);

            sgemm(Order, TransA, TransB, M, N, K, alpha, fA, lda, fB, ldb, beta, fC, ldc);

            C.assign(fC);
        /*} else {
            // TODO: uncomment this once we have optimized gemm calls
            var t = MMulTranspose.builder()
                    .transposeA(false)
                    .transposeB(false)
                    .transposeResult(false)
                    .build();
            var op = new Mmul(A, B, C, t);
            Nd4j.exec(op);
        }
         */
    }

    @Override
    protected void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda,
                    INDArray B, int ldb, float beta, INDArray C, int ldc) {
        if (!Nd4j.isFallbackModeEnabled()) {
            Nd4j.getBlasLapackDelegator(). cblas_sgemm(convertOrder('f'), convertTranspose(TransA), convertTranspose(TransB), M, N, K, alpha,
                            (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(),
                            ldb, beta, (FloatPointer) C.data().addressPointer(), ldc);
        } else {
            Nd4j.getExecutioner()
                            .exec(new AggregateGEMM('f', TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
        }
    }

    @Override
    protected void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda, INDArray B,
                    int ldb, float beta, INDArray C, int ldc) {
        Nd4j.getBlasLapackDelegator(). cblas_ssymm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), M, N, alpha,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(), ldb,
                        beta, (FloatPointer) C.data().addressPointer(), ldc);
    }

    @Override
    protected void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, float beta,
                    INDArray C, int ldc) {
        Nd4j.getBlasLapackDelegator().cblas_ssyrk(convertOrder('f'), convertUplo(Uplo), convertTranspose(Trans), N, K, alpha,
                        (FloatPointer) A.data().addressPointer(), lda, beta, (FloatPointer) C.data().addressPointer(),
                        ldc);
    }

    @Override
    protected void ssyr2k(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, INDArray B,
                    int ldb, float beta, INDArray C, int ldc) {
        Nd4j.getBlasLapackDelegator(). cblas_ssyr2k(convertOrder('f'), convertUplo(Uplo), convertTranspose(Trans), N, K, alpha,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(), ldb,
                        beta, (FloatPointer) C.data().addressPointer(), ldc);
    }

    @Override
    protected void strmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        Nd4j.getBlasLapackDelegator(). cblas_strmm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), convertTranspose(TransA), Diag, M, N,
                        alpha, (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(),
                        ldb);
    }

    @Override
    protected void strsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        Nd4j.getBlasLapackDelegator(). cblas_strsm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), convertTranspose(TransA), Diag, M, N,
                        alpha, (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(),
                        ldb);
    }

    @Override
    protected void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A, int lda,
                    INDArray B, int ldb, double beta, INDArray C, int ldc) {
        if (!Nd4j.isFallbackModeEnabled()) {
            Nd4j.getBlasLapackDelegator().cblas_dgemm(convertOrder('f'), convertTranspose(TransA), convertTranspose(TransB), M, N, K, alpha,
                            (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) B.data().addressPointer(),
                            ldb, beta, (DoublePointer) C.data().addressPointer(), ldc);
        } else {
            Nd4j.getExecutioner()
                            .exec(new AggregateGEMM('f', TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
        }
    }

    @Override
    protected void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda, INDArray B,
                    int ldb, double beta, INDArray C, int ldc) {
        Nd4j.getBlasLapackDelegator().cblas_dsymm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), M, N, alpha,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) B.data().addressPointer(), ldb,
                        beta, (DoublePointer) C.data().addressPointer(), ldc);
    }

    @Override
    protected void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda,
                    double beta, INDArray C, int ldc) {
        Nd4j.getBlasLapackDelegator().cblas_dsyrk(convertOrder('f'), convertUplo(Uplo), convertTranspose(Trans), N, K, alpha,
                        (DoublePointer) A.data().addressPointer(), lda, beta, (DoublePointer) C.data().addressPointer(),
                        ldc);
    }

    @Override
    protected void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda,
                    INDArray B, int ldb, double beta, INDArray C, int ldc) {
        Nd4j.getBlasLapackDelegator().cblas_dsyr2k(convertOrder('f'), convertUplo(Uplo), convertTranspose(Trans), N, K, alpha,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) B.data().addressPointer(), ldb,
                        beta, (DoublePointer) C.data().addressPointer(), ldc);
    }

    @Override
    protected void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        Nd4j.getBlasLapackDelegator().cblas_dtrmm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), convertTranspose(TransA), Diag, M, N,
                        alpha, (DoublePointer) A.data().addressPointer(), lda,
                        (DoublePointer) B.data().addressPointer(), ldb);
    }

    @Override
    protected void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        Nd4j.getBlasLapackDelegator(). cblas_dtrsm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), convertTranspose(TransA), Diag, M, N,
                        alpha, (DoublePointer) A.data().addressPointer(), lda,
                        (DoublePointer) B.data().addressPointer(), ldb);
    }
}
