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

package org.nd4j.linalg.api.blas;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * Level 2 blas implementations.
 * Incx and other parameters are inferred
 * from the given ndarrays.
 *
 * To avoid boxing, doubles are used in place of normal numbers.
 * The underlying implementation will call the proper data opType.
 *
 * This is a fortran 95 style api that gives us the efficiency
 * and flexibility of the fortran 77 api
 *
 * Credit to:
 * https://www.ualberta.ca/AICT/RESEARCH/LinuxClusters/doc/mkl81/mklqref/blaslev2.htm
 *
 * for the descriptions
 *
 * @author Adam Gibson
 */
public interface Level2 {
    /**
     * gemv computes a matrix-vector product using a general matrix and performs one of the following matrix-vector operations: 
     y := alpha*a*x + beta*y  for trans = 'N'or'n'; 
     y := alpha*a'*x + beta*y  for trans = 'T'or't'; 
     y := alpha*conjg(a')*x + beta*y  for trans = 'C'or'c'. 
     Here a is an m-by-n band matrix, x and y are vectors, alpha and beta are scalars.
     * @param order
     * @param transA
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    void gemv(char order, char transA, double alpha, INDArray A, INDArray X, double beta, INDArray Y);

    /**
     * gbmv computes a matrix-vector product using a general band matrix and performs one of the following matrix-vector operations:
     y := alpha*a*x + beta*y  for trans = 'N'or'n';
     y := alpha*a'*x + beta*y  for trans = 'T'or't';
     y := alpha*conjg(a')*x + beta*y  for trans = 'C'or'c'.
     Here a is an m-by-n band matrix with ku superdiagonals and kl subdiagonals, x and y are vectors, alpha and beta are scalars.
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
    void gbmv(char order, char TransA, int KL, int KU, double alpha, INDArray A, INDArray X, double beta, INDArray Y);


    /**
     *  performs a rank-1 update of a general m-by-n matrix a:
     a := alpha*x*y' + a.
     * @param order
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    void ger(char order, double alpha, INDArray X, INDArray Y, INDArray A);

    /**
     * sbmv computes a matrix-vector product using a symmetric band matrix:
     y := alpha*a*x + beta*y.
     Here a is an n-by-n symmetric band matrix with k superdiagonals, x and y are n-element vectors, alpha and beta are scalars.
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    void sbmv(char order, char Uplo, double alpha, INDArray A, INDArray X, double beta, INDArray Y);

    /**
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param Ap
     * @param X
     * @param beta
     * @param Y
     */
    void spmv(char order, char Uplo, double alpha, INDArray Ap, INDArray X, double beta, INDArray Y);

    /**
     * spr performs a rank-1 update of an n-by-n packed symmetric matrix a:
     a := alpha*x*x' + a. 
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Ap
     */
    void spr(char order, char Uplo, double alpha, INDArray X, INDArray Ap);

    /**
     * ?spr2 performs a rank-2 update of an n-by-n packed symmetric matrix a:
     a := alpha*x*y' + alpha*y*x' + a. 
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    void spr2(char order, char Uplo, double alpha, INDArray X, INDArray Y, INDArray A);

    /**
     * symv computes a matrix-vector product for a symmetric matrix:
     y := alpha*a*x + beta*y.
     Here a is an n-by-n symmetric matrix; x and y are n-element vectors, alpha and beta are scalars.
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    void symv(char order, char Uplo, double alpha, INDArray A, INDArray X, double beta, INDArray Y);

    /**
     * syr performs a rank-1 update of an n-by-n symmetric matrix a:
     a := alpha*x*x' + a.
     * @param order
     * @param Uplo
     * @param N
     * @param alpha
     * @param X
     * @param A
     */
    void syr(char order, char Uplo, int N, double alpha, INDArray X, INDArray A);

    /**
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    void syr2(char order, char Uplo, double alpha, INDArray X, INDArray Y, INDArray A);

    /**
     * syr2 performs a rank-2 update of an n-by-n symmetric matrix a:
     a := alpha*x*y' + alpha*y*x' + a.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    void tbmv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X);

    /**
     * ?tbsv solves a system of linear equations whose coefficients are in a triangular band matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    void tbsv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X);

    /**
     * tpmv computes a matrix-vector product using a triangular packed matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param Ap
     * @param X
     */
    void tpmv(char order, char Uplo, char TransA, char Diag, INDArray Ap, INDArray X);

    /**
     * tpsv solves a system of linear equations whose coefficients are in a triangular packed matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param Ap
     * @param X
     */
    void tpsv(char order, char Uplo, char TransA, char Diag, INDArray Ap, INDArray X);

    /**
     * trmv computes a matrix-vector product using a triangular matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    void trmv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X);

    /**
     * trsv solves a system of linear equations whose coefficients are in a triangular matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    void trsv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X);
}
