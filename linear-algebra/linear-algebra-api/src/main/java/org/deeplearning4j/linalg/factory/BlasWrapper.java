// --- BEGIN LICENSE BLOCK ---
/* 
 * Copyright (c) 2009-2011, Mikio L. Braun
 *               2011, Nicolas Oury
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 * 
 *     * Neither the name of the Technische Universität Berlin nor the
 *       names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// --- END LICENSE BLOCK ---

package org.deeplearning4j.linalg.factory;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;



/**
 * This class provides a cleaner direct interface to the BLAS routines by
 * extracting the parameters of the matrices from the matrices itself.
 * <p/>
 * For example, you can just pass the vector and do not have to pass the length,
 * corresponding DoubleBuffer, offset and step size explicitly.
 * <p/>
 * Currently, all the general matrix routines are implemented.
 */
public interface BlasWrapper<NDARRAY_TYPE extends INDArray,COMPLEX_NDARRAY_TYPE extends IComplexNDArray,NUMBER_TYPE extends IComplexNumber> {
    /***************************************************************************
     * BLAS Level 1
     */

    /**
     * Compute x <-> y (swap two matrices)
     */
    public NDARRAY_TYPE swap(NDARRAY_TYPE x, NDARRAY_TYPE y);

    /**
     * Compute x <- alpha * x (scale a matrix)
     */
    public NDARRAY_TYPE scal(double alpha, NDARRAY_TYPE x);

    public COMPLEX_NDARRAY_TYPE scal(NUMBER_TYPE alpha, COMPLEX_NDARRAY_TYPE x);

    /**
     * Compute y <- x (copy a matrix)
     */
    public NDARRAY_TYPE copy(NDARRAY_TYPE x, NDARRAY_TYPE y);

    public COMPLEX_NDARRAY_TYPE copy(COMPLEX_NDARRAY_TYPE x, COMPLEX_NDARRAY_TYPE y);

    /**
     * Compute y <- alpha * x + y (elementwise addition)
     */
    public NDARRAY_TYPE axpy(double da, NDARRAY_TYPE dx, NDARRAY_TYPE dy);

    public COMPLEX_NDARRAY_TYPE axpy(NUMBER_TYPE da, COMPLEX_NDARRAY_TYPE dx, COMPLEX_NDARRAY_TYPE dy);

    /**
     * Compute x^T * y (dot product)
     */
    public double dot(NDARRAY_TYPE x, NDARRAY_TYPE y);
    /**
     * Compute x^T * y (dot product)
     */
    public NUMBER_TYPE dotc(COMPLEX_NDARRAY_TYPE x, COMPLEX_NDARRAY_TYPE y);
    /**
     * Compute x^T * y (dot product)
     */
    public NUMBER_TYPE dotu(COMPLEX_NDARRAY_TYPE x, COMPLEX_NDARRAY_TYPE y);
    /**
     * Compute || x ||_2 (2-norm)
     */
    public double nrm2(NDARRAY_TYPE x);

    public double nrm2(COMPLEX_NDARRAY_TYPE x);

    /**
     * Compute || x ||_1 (1-norm, sum of absolute values)
     */
    public double asum(NDARRAY_TYPE x);

    public double asum(COMPLEX_NDARRAY_TYPE x);

    /**
     * Compute index of element with largest absolute value (index of absolute
     * value maximum)
     */
    public int iamax(NDARRAY_TYPE x);

    /**
     * Compute index of element with largest absolute value (complex version).
     *
     * @param x matrix
     * @return index of element with largest absolute value.
     */
    public int iamax(COMPLEX_NDARRAY_TYPE x);
    /***************************************************************************
     * BLAS Level 2
     */

    /**
     * Compute y <- alpha*op(a)*x + beta * y (general matrix vector
     * multiplication)
     */
    public NDARRAY_TYPE gemv(double alpha, NDARRAY_TYPE a,
                         NDARRAY_TYPE x, double beta, NDARRAY_TYPE y);

    /**
     * Compute A <- alpha * x * y^T + A (general rank-1 update)
     */
    public NDARRAY_TYPE ger(double alpha, NDARRAY_TYPE x,
                        NDARRAY_TYPE y, NDARRAY_TYPE a);

    /**
     * Compute A <- alpha * x * y^T + A (general rank-1 update)
     */
    public COMPLEX_NDARRAY_TYPE geru(NUMBER_TYPE alpha, COMPLEX_NDARRAY_TYPE x,
                                COMPLEX_NDARRAY_TYPE y, COMPLEX_NDARRAY_TYPE a);
    /**
     * Compute A <- alpha * x * y^H + A (general rank-1 update)
     */
    public COMPLEX_NDARRAY_TYPE gerc(NUMBER_TYPE alpha, COMPLEX_NDARRAY_TYPE x,
                                COMPLEX_NDARRAY_TYPE y, COMPLEX_NDARRAY_TYPE a);
    /***************************************************************************
     * BLAS Level 3
     */

    /**
     * Compute c <- a*b + beta * c (general matrix matrix
     * multiplication)
     */
    public NDARRAY_TYPE gemm(double alpha, NDARRAY_TYPE a,
                         NDARRAY_TYPE b, double beta, NDARRAY_TYPE c);
    public COMPLEX_NDARRAY_TYPE gemm(NUMBER_TYPE alpha, COMPLEX_NDARRAY_TYPE a,
                                COMPLEX_NDARRAY_TYPE b, NUMBER_TYPE beta, COMPLEX_NDARRAY_TYPE c);
    /***************************************************************************
     * LAPACK
     */

    public NDARRAY_TYPE gesv(NDARRAY_TYPE a, int[] ipiv,
                         NDARRAY_TYPE b);

//STOP

    void checkInfo(String name, int info);
//START

    public NDARRAY_TYPE sysv(char uplo, NDARRAY_TYPE a, int[] ipiv,
                         NDARRAY_TYPE b);

    public int syev(char jobz, char uplo, NDARRAY_TYPE a, NDARRAY_TYPE w);
    public int syevx(char jobz, char range, char uplo, NDARRAY_TYPE a,
                     double vl, double vu, int il, int iu, double abstol,
                     NDARRAY_TYPE w, NDARRAY_TYPE z);

    public int syevd(char jobz, char uplo, NDARRAY_TYPE A,
                     NDARRAY_TYPE w);

    public int syevr(char jobz, char range, char uplo, NDARRAY_TYPE a,
                     double vl, double vu, int il, int iu, double abstol,
                     NDARRAY_TYPE w, NDARRAY_TYPE z, int[] isuppz);

    public void posv(char uplo, NDARRAY_TYPE A, NDARRAY_TYPE B);

    public int geev(char jobvl, char jobvr, NDARRAY_TYPE A,
                    NDARRAY_TYPE WR, NDARRAY_TYPE WI, NDARRAY_TYPE VL, NDARRAY_TYPE VR);

    public int sygvd(int itype, char jobz, char uplo, NDARRAY_TYPE A, NDARRAY_TYPE B, NDARRAY_TYPE W);

    /**
     * Generalized Least Squares via *GELSD.
     *
     * Note that B must be padded to contain the solution matrix. This occurs when A has fewer rows
     * than columns.
     *
     * For example: in A * X = B, A is (m,n), X is (n,k) and B is (m,k). Now if m < n, since B is overwritten to contain
     * the solution (in classical LAPACK style), B needs to be padded to be an (n,k) matrix.
     *
     * Likewise, if m > n, the solution consists only of the first n rows of B.
     *
     * @param A an (m,n) matrix
     * @param B an (max(m,n), k) matrix (well, at least)
     */
    public void gelsd(NDARRAY_TYPE A, NDARRAY_TYPE B);

    public void geqrf(NDARRAY_TYPE A, NDARRAY_TYPE tau);

    public  void ormqr(char side, char trans, NDARRAY_TYPE A, NDARRAY_TYPE tau, NDARRAY_TYPE C);


    public  void dcopy(int n, double[] dx, int dxIdx, int incx, double[] dy, int dyIdx, int incy);






}
