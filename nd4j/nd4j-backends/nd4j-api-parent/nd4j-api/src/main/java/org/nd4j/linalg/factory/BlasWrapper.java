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

// --- BEGIN LICENSE BLOCK ---
// --- END LICENSE BLOCK ---

package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.blas.Lapack;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.blas.Level2;
import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * This class provides a cleaner direct interface to the BLAS routines by extracting the parameters of the matrices from
 * the matrices itself.
 * <p/>
 * For example, you can just pass the vector and do not have to pass the length, corresponding DoubleBuffer, offset and
 * step size explicitly.
 * <p/>
 * Currently, all the general matrix routines are implemented.
 */
public interface BlasWrapper {
    /***************************************************************************
     * BLAS Level 1
     */

    /**
     * Compute x <-> y (swap two matrices)
     */
    INDArray swap(INDArray x, INDArray y);

    /**
     * Return the level 1 functions
     * for this blas impl
     * @return
     */
    Level1 level1();

    /**
     * Return the level 2 functions
     * for this blas impl
     * @return
     */
    Level2 level2();

    /**
     * Return the level 3 functions
     * for this blas impl
     * @return
     */
    Level3 level3();

    /**
     * LAPack interface
     * @return
     */
    Lapack lapack();


    @Deprecated
    INDArray scal(double alpha, INDArray x);

    /**
     * Compute x <- alpha * x (scale a matrix)
     */
    @Deprecated
    INDArray scal(float alpha, INDArray x);


    /**
     * Compute y <- x (copy a matrix)
     */
    INDArray copy(INDArray x, INDArray y);

    @Deprecated
    INDArray axpy(double da, INDArray dx, INDArray dy);

    /**
     * Compute y <- alpha * x + y (elementwise addition)
     */
    @Deprecated
    INDArray axpy(float da, INDArray dx, INDArray dy);

    /**
     * Compute y <- y + x * alpha
     * @param da the alpha to multiply by
     * @param dx
     * @param dy
     * @return
     */
    INDArray axpy(Number da, INDArray dx, INDArray dy);

    /**
     * Compute x^T * y (dot product)
     */
    double dot(INDArray x, INDArray y);

    /**
     * Compute || x ||_2 (2-norm)
     */
    double nrm2(INDArray x);

    /**
     * Compute || x ||_1 (1-norm, sum of absolute values)
     */
    double asum(INDArray x);

    /**
     * Compute index of element with largest absolute value (index of absolute
     * value maximum)
     */
    int iamax(INDArray x);

    /**
     * ************************************************************************
     * BLAS Level 2
     */

    INDArray gemv(Number alpha, INDArray a, INDArray x, double beta, INDArray y);

    @Deprecated
    INDArray gemv(double alpha, INDArray a, INDArray x, double beta, INDArray y);

    /**
     * Compute y <- alpha*op(a)*x + beta * y (general matrix vector
     * multiplication)
     */
    @Deprecated
    INDArray gemv(float alpha, INDArray a, INDArray x, float beta, INDArray y);

    INDArray ger(Number alpha, INDArray x, INDArray y, INDArray a);

    @Deprecated
    INDArray ger(double alpha, INDArray x, INDArray y, INDArray a);

    /**
     * Compute A <- alpha * x * y^T + A (general rank-1 update)
     */
    INDArray ger(float alpha, INDArray x, INDArray y, INDArray a);

    /**
     * ************************************************************************
     * BLAS Level 3
     */
    @Deprecated
    INDArray gemm(double alpha, INDArray a, INDArray b, double beta, INDArray c);

    /**
     * Compute c <- a*b + beta * c (general matrix matrix
     * multiplication)
     */
    @Deprecated
    INDArray gemm(float alpha, INDArray a, INDArray b, float beta, INDArray c);


    /**
     * ************************************************************************
     * LAPACK
     */

    INDArray gesv(INDArray a, int[] ipiv, INDArray b);

    //STOP

    void checkInfo(String name, int info);
    //START

    INDArray sysv(char uplo, INDArray a, int[] ipiv, INDArray b);



    int syev(char jobz, char uplo, INDArray a, INDArray w);

    int syevx(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol,
                    INDArray w, INDArray z);

    int syevx(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol,
                    INDArray w, INDArray z);

    int syevd(char jobz, char uplo, INDArray A, INDArray w);

    @Deprecated
    int syevr(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol,
                    INDArray w, INDArray z, int[] isuppz);

    @Deprecated
    int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol,
                    INDArray w, INDArray z, int[] isuppz);


    int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, Number abstol,
                    INDArray w, INDArray z, int[] isuppz);


    void posv(char uplo, INDArray A, INDArray B);

    int geev(char jobvl, char jobvr, INDArray A, INDArray WR, INDArray WI, INDArray VL, INDArray VR);

    int sygvd(int itype, char jobz, char uplo, INDArray A, INDArray B, INDArray W);

    /**
     * Generalized Least Squares via *GELSD.
     * <p/>
     * Note that B must be padded to contain the solution matrix. This occurs when A has fewer rows
     * than columns.
     * <p/>
     * For example: in A * X = B, A is (m,n), X is (n,k) and B is (m,k). Now if m < n, since B is overwritten to contain
     * the solution (in classical LAPACK style), B needs to be padded to be an (n,k) matrix.
     * <p/>
     * Likewise, if m > n, the solution consists only of the first n rows of B.
     *
     * @param A an (m,n) matrix
     * @param B an (max(m,n), k) matrix (well, at least)
     */
    void gelsd(INDArray A, INDArray B);

    void geqrf(INDArray A, INDArray tau);

    void ormqr(char side, char trans, INDArray A, INDArray tau, INDArray C);


    @Deprecated
    void saxpy(double alpha, INDArray x, INDArray y);

    /**
     * Abstraction over saxpy
     *
     * @param alpha the alpha to scale by
     * @param x     the ndarray to use
     * @param y     the ndarray to use
     */
    @Deprecated
    void saxpy(float alpha, INDArray x, INDArray y);


}
