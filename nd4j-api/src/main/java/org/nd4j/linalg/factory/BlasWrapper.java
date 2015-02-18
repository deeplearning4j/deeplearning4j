/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

// --- BEGIN LICENSE BLOCK ---
// --- END LICENSE BLOCK ---

package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * This class provides a cleaner direct interface to the BLAS routines by
 * extracting the parameters of the matrices from the matrices itself.
 * <p/>
 * For example, you can just pass the vector and do not have to pass the length,
 * corresponding DoubleBuffer, offset and step size explicitly.
 * <p/>
 * Currently, all the general matrix routines are implemented.
 */
public interface BlasWrapper<NDARRAY_TYPE extends INDArray> {
    /***************************************************************************
     * BLAS Level 1
     */

    /**
     * Compute x <-> y (swap two matrices)
     */
    public NDARRAY_TYPE swap(NDARRAY_TYPE x, NDARRAY_TYPE y);

    INDArray scal(double alpha, INDArray x);

    /**
     * Compute x <- alpha * x (scale a matrix)
     */
    public NDARRAY_TYPE scal(float alpha, NDARRAY_TYPE x);

    public IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x);


    IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x);

    /**
     * Compute y <- x (copy a matrix)
     */
    public NDARRAY_TYPE copy(NDARRAY_TYPE x, NDARRAY_TYPE y);

    public IComplexNDArray copy(IComplexNDArray x, IComplexNDArray y);

    INDArray axpy(double da, INDArray dx, INDArray dy);

    /**
     * Compute y <- alpha * x + y (elementwise addition)
     */
    public NDARRAY_TYPE axpy(float da, NDARRAY_TYPE dx, NDARRAY_TYPE dy);

    public IComplexNDArray axpy(IComplexNumber da, IComplexNDArray dx, IComplexNDArray dy);

    /**
     * Compute x^T * y (dot product)
     */
    public double dot(NDARRAY_TYPE x, NDARRAY_TYPE y);

    /**
     * Compute x^T * y (dot product)
     */
    public IComplexNumber dotc(IComplexNDArray x, IComplexNDArray y);

    /**
     * Compute x^T * y (dot product)
     */
    public IComplexNumber dotu(IComplexNDArray x, IComplexNDArray y);

    /**
     * Compute || x ||_2 (2-norm)
     */
    public double nrm2(NDARRAY_TYPE x);

    public double nrm2(IComplexNDArray x);

    /**
     * Compute || x ||_1 (1-norm, sum of absolute values)
     */
    public double asum(NDARRAY_TYPE x);

    public double asum(IComplexNDArray x);

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
    public int iamax(IComplexNDArray x);

    /**
     * ************************************************************************
     * BLAS Level 2
     */

    INDArray gemv(double alpha, INDArray a,
                  INDArray x, double beta, INDArray y);

    /**
     * Compute y <- alpha*op(a)*x + beta * y (general matrix vector
     * multiplication)
     */
    public NDARRAY_TYPE gemv(float alpha, NDARRAY_TYPE a,
                             NDARRAY_TYPE x, float beta, NDARRAY_TYPE y);

    INDArray ger(double alpha, INDArray x,
                 INDArray y, INDArray a);

    /**
     * Compute A <- alpha * x * y^T + A (general rank-1 update)
     */
    public NDARRAY_TYPE ger(float alpha, NDARRAY_TYPE x,
                            NDARRAY_TYPE y, NDARRAY_TYPE a);

    IComplexNDArray gemv(IComplexDouble alpha, IComplexNDArray a,
                         IComplexNDArray x, IComplexDouble beta, IComplexNDArray y);

    IComplexNDArray gemv(IComplexFloat alpha, IComplexNDArray a, IComplexNDArray x, IComplexFloat beta, IComplexNDArray y);

    IComplexNDArray geru(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a);

    /**
     * Compute A <- alpha * x * y^T + A (general rank-1 update)
     */
    public IComplexNDArray geru(IComplexFloat alpha, IComplexNDArray x,
                                IComplexNDArray y, IComplexNDArray a);

    /**
     * Compute A <- alpha * x * y^H + A (general rank-1 update)
     */
    public IComplexNDArray gerc(IComplexFloat alpha, IComplexNDArray x,
                                IComplexNDArray y, IComplexNDArray a);

    IComplexNDArray gerc(IComplexDouble alpha, IComplexNDArray x,
                         IComplexNDArray y, IComplexNDArray a);

    /**
     * ************************************************************************
     * BLAS Level 3
     */

    INDArray gemm(double alpha, INDArray a,
                  INDArray b, double beta, INDArray c);

    /**
     * Compute c <- a*b + beta * c (general matrix matrix
     * multiplication)
     */
    public NDARRAY_TYPE gemm(float alpha, NDARRAY_TYPE a,
                             NDARRAY_TYPE b, float beta, NDARRAY_TYPE c);

    public IComplexNDArray gemm(IComplexNumber alpha, IComplexNDArray a,
                                IComplexNDArray b, IComplexNumber beta, IComplexNDArray c);


    /**
     * ************************************************************************
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

    int syevx(char jobz, char range, char uplo, INDArray a,
              double vl, double vu, int il, int iu, double abstol,
              INDArray w, INDArray z);

    public int syevx(char jobz, char range, char uplo, NDARRAY_TYPE a,
                     float vl, float vu, int il, int iu, float abstol,
                     NDARRAY_TYPE w, NDARRAY_TYPE z);

    public int syevd(char jobz, char uplo, NDARRAY_TYPE A,
                     NDARRAY_TYPE w);

    int syevr(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z, int[] isuppz);

    public int syevr(char jobz, char range, char uplo, NDARRAY_TYPE a,
                     float vl, float vu, int il, int iu, float abstol,
                     NDARRAY_TYPE w, NDARRAY_TYPE z, int[] isuppz);

    public void posv(char uplo, NDARRAY_TYPE A, NDARRAY_TYPE B);

    public int geev(char jobvl, char jobvr, NDARRAY_TYPE A,
                    NDARRAY_TYPE WR, NDARRAY_TYPE WI, NDARRAY_TYPE VL, NDARRAY_TYPE VR);

    public int sygvd(int itype, char jobz, char uplo, NDARRAY_TYPE A, NDARRAY_TYPE B, NDARRAY_TYPE W);

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
    public void gelsd(NDARRAY_TYPE A, NDARRAY_TYPE B);

    public void geqrf(NDARRAY_TYPE A, NDARRAY_TYPE tau);

    public void ormqr(char side, char trans, NDARRAY_TYPE A, NDARRAY_TYPE tau, NDARRAY_TYPE C);


    public void dcopy(int n, float[] dx, int dxIdx, int incx, float[] dy, int dyIdx, int incy);

    void saxpy(double alpha, INDArray x, INDArray y);

    /**
     * Abstraction over saxpy
     *
     * @param alpha the alpha to scale by
     * @param x     the ndarray to use
     * @param y     the ndarray to use
     */
    void saxpy(float alpha, INDArray x, INDArray y);


}
