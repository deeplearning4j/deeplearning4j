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
 * Lapack interface
 *
 * @author Adam Gibson
 * @author rcorbish
 */
public interface Lapack {

    /**
     * LU decomposiiton of a matrix
     * Factorize a matrix A
     *
     * The matrix A is overridden by the L & U combined.
     * The permutation results are returned directly as a vector. To 
     * create the permutation matrix use getPFactor method
     * To split out the L & U matrix use getLFactor and getUFactor methods
     *
     * getrf = triangular factorization (TRF) of a general matrix (GE)
     *
     * @param A the input matrix, it will be overwritten with the factors
     * @returns Permutation array
     * @throws Error - with a message to indicate failure (usu. bad params)
     */
    INDArray getrf(INDArray A);



    /**
     * QR decomposiiton of a matrix
     * Factorize a matrix A such that A = QR
     *
     * The matrix A is overwritten by the Q component (i.e. destroyed)
     *
     * geqrf = QR factorization of a general matrix (GE) into an orthogonal
     *         matrix Q and an upper triangular R matrix
     *
     * @param A the input matrix, it will be overwritten with the factors
     * @param The R array if null R is not returned
     * @throws Error - with a message to indicate failure (usu. bad params)
     */
    void geqrf(INDArray A, INDArray R);



    /**
     * Triangular decomposiiton of a positive definite matrix ( cholesky )
     * Factorize a matrix A such that A = LL* (assuming lower==true) or
     * A = U*U   (a * represents conjugate i.e. if matrix is real U* is a transpose) 
     *
     * The matrix A is overridden by the L (or U).
     *
     * potrf = LU factorization of a positive definite matrix (PO) into a
     *         lower L ( or upper U ) triangular matrix
     *
     * @param A the input matrix, it will be overwritten with the factors
     * @param whether to return the upper (false) or lower factor
     * @returns Permutation array
     * @throws Error - with a message to indicate failure (usu. bad params)
     */
    void potrf(INDArray A, boolean lower);


    /**
     * Caclulate the eigenvalues and vectors of a symmetric matrix. The
     * symmetric matrix means the results are guaranteed to be real (not complex)
     *
     * The matrix A is overridden by the eigenvectors. The eigenvalues
     * are returned separately
     *
     * @param A the input matrix, it will be overwritten with the eigenvectors
     * @param V the resulting eigenvalues
     * @throws Error - with a message to indicate failure (usu. bad params)
     */
    int syev(char jobz, char uplo, INDArray A, INDArray V);



    /**
     * SVD decomposiiton of a matrix
     * Factorize a matrix into its singular vectors and eigenvalues
     * The decomposition is such that:
     *
     * 		A = U x S x VT
     *
     * gesvd = singular value decomposition (SVD) of a general matrix (GE)
     *
     * @param A the input matrix
     * @param S the eigenvalues as a vector
     * @param U the left singular vectors as a matrix. Maybe null if no S required
     * @param VT the right singular vectors as a (transposed) matrix. Maybe null if no V required
     * @throws Error - with a message to indicate failure (usu. bad params)
     */
    void gesvd(INDArray A, INDArray S, INDArray U, INDArray VT);



    /** 
    * This method takes one of the ipiv returns from LAPACK and creates
    * the permutation matrix. When factorizing, it is useful to avoid underflows
    * and overflows by reordering rows/and or columns of the input matrix (mostly
    * these methods solve simultaneous equations, so order is not important). 
    * The ipiv method assumes that only row ordering is done ( never seen column 
    * ordering done )
    *
    * @param M - the size of the permutation matrix ( usu. the # rows in factored matrix )
    * @param ipiv - the vector returned from a refactoring
    * @returned the square permutation matrix - size is the M x M
    */
    INDArray getPFactor(int M, INDArray ipiv);


    /**
    * extracts the L (lower triangular) matrix from the LU factor result
    * L will be the same dimensions as A
    *
    * @param A - the combined L & U matrices returned from factorization
    * @returned the lower triangular with unit diagonal
    */
    INDArray getLFactor(INDArray A);


    /**
    * extracts the U (upper triangular) matrix from the LU factor result
    * U will be n x n matrix where n = num cols in A
    *
    * @param A - the combined L & U matrices returned from factorization
    * @returned the upper triangular matrix
    */
    INDArray getUFactor(INDArray A);


    // generate inverse of a matrix given its LU decomposition

    /**
     * Generate inverse ggiven LU decomp
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param WORK
     * @param lwork
     * @param INFO
     */
    void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO);

}
