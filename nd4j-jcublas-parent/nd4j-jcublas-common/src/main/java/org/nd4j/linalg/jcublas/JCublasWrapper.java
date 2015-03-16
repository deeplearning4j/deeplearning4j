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

package org.nd4j.linalg.jcublas;

import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Blas wrapper for JCUDA
 *
 * @author mjk
 * @author Adam Gibson
 */
public class JCublasWrapper implements org.nd4j.linalg.factory.BlasWrapper {
    @Override
    public INDArray swap(INDArray x, INDArray y) {
        SimpleJCublas.swap(x, y);
        return y;
    }

    @Override
    public INDArray scal(double alpha, INDArray x) {
        return SimpleJCublas.scal(alpha, x);

    }

    @Override
    public INDArray scal(float alpha, INDArray x) {
        SimpleJCublas.scal(alpha, x);
        return x;
    }


    @Override
    public IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) {
        return SimpleJCublas.scal(alpha, x);

    }

    @Override
    public IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) {
        return SimpleJCublas.scal(alpha, x);

    }

    @Override
    public INDArray copy(INDArray x, INDArray y) {
        SimpleJCublas.copy(x, y);
        return y;
    }

    @Override
    public IComplexNDArray copy(IComplexNDArray x, IComplexNDArray y) {
        SimpleJCublas.copy(x, y);
        return y;
    }

    @Override
    public INDArray axpy(double da, INDArray dx, INDArray dy) {
        SimpleJCublas.axpy(da, dx, dy);
        return dy;
    }

    @Override
    public INDArray axpy(float da, INDArray dx, INDArray dy) {
        SimpleJCublas.axpy(da, dx, dy);
        return dy;
    }


    @Override
    public IComplexNDArray axpy(IComplexNumber da, IComplexNDArray dx, IComplexNDArray dy) {
        if (da instanceof IComplexDouble) {
            SimpleJCublas.axpy((IComplexDouble) da, dx, dy);

        } else
            SimpleJCublas.axpy((IComplexFloat) da, dx, dy);

        return dy;
    }

    public double dot(INDArray x, INDArray y) {

        return SimpleJCublas.dot(x, y);
    }

    //@Override
    public double dotd(INDArray x, INDArray y) {
        return SimpleJCublas.dot(x, y);
    }

    @Override
    public IComplexNumber dotc(IComplexNDArray x, IComplexNDArray y) {
        return SimpleJCublas.dot(x, y);
    }

    @Override
    public IComplexNumber dotu(IComplexNDArray x, IComplexNDArray y) {
        return SimpleJCublas.dotu(x, y);
    }

    @Override
    public double nrm2(INDArray x) {
        return SimpleJCublas.nrm2(x);
    }

    @Override
    public double nrm2(IComplexNDArray x) {
        return SimpleJCublas.nrm2(x);
    }

    @Override
    public double asum(INDArray x) {
        return SimpleJCublas.asum(x);

    }

    @Override
    public double asum(IComplexNDArray x) {
        return SimpleJCublas.asum(x);
    }

    @Override
    public int iamax(INDArray x) {
        return SimpleJCublas.iamax(x);
    }

    @Override
    public int iamax(IComplexNDArray x) {
        return SimpleJCublas.iamax(x);
    }

    @Override
    public INDArray gemv(double alpha, INDArray a, INDArray x, double beta, INDArray y) {
        SimpleJCublas.gemv(a, x, y, alpha, beta);
        return y;
    }

    @Override
    public INDArray gemv(float alpha, INDArray a, INDArray x, float beta, INDArray y) {
        return SimpleJCublas.gemv(a, x, y, alpha, beta);
    }

    @Override
    public INDArray ger(double alpha, INDArray x, INDArray y, INDArray a) {
        return SimpleJCublas.ger(x, y, a, alpha);
    }

    @Override
    public INDArray ger(float alpha, INDArray x, INDArray y, INDArray a) {
        return SimpleJCublas.ger(x, y, a, alpha);
    }

    @Override
    public IComplexNDArray gemv(IComplexDouble alpha, IComplexNDArray a, IComplexNDArray x, IComplexDouble beta, IComplexNDArray y) {
        return SimpleJCublas.gemv(a, x, alpha, y, beta);

    }

    @Override
    public IComplexNDArray gemv(IComplexFloat alpha, IComplexNDArray a, IComplexNDArray x, IComplexFloat beta, IComplexNDArray y) {
        return SimpleJCublas.gemv(a, x, alpha, y, beta);
    }

    @Override
    public IComplexNDArray geru(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        throw new UnsupportedOperationException();
    }


    @Override
    public IComplexNDArray geru(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleJCublas.geru(x, y, a, alpha.asDouble());
    }

    @Override
    public IComplexNDArray gerc(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleJCublas.gerc(x, y, a, alpha.asDouble());
    }

    @Override
    public IComplexNDArray gerc(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray gemm(double alpha, INDArray a, INDArray b, double beta, INDArray c) {
        return SimpleJCublas.gemm(a, b, c, alpha, beta);
    }

    @Override
    public INDArray gemm(float alpha, INDArray a, INDArray b, float beta, INDArray c) {
        return SimpleJCublas.gemm(a, b, c, alpha, beta);
    }


    @Override
    public IComplexNDArray gemm(IComplexNumber alpha, IComplexNDArray a, IComplexNDArray b, IComplexNumber beta, IComplexNDArray c) {
        if (beta instanceof IComplexDouble)
            SimpleJCublas.gemm(a, b, alpha.asDouble(), c, beta.asDouble());
        else
            SimpleJCublas.gemm(a, b, alpha.asFloat(), c, beta.asFloat());

        return c;
    }

    @Override
    public INDArray gesv(INDArray a, int[] ipiv, INDArray b) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void checkInfo(String name, int info) {
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray sysv(char uplo, INDArray a, int[] ipiv, INDArray b) {
        throw new UnsupportedOperationException();

    }

    @Override
    public int syev(char jobz, char uplo, INDArray a, INDArray w) {
        return 0;
    }

    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z) {
        throw new UnsupportedOperationException();

    }

    //@Override
    public int syevx(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z) {
        throw new UnsupportedOperationException();

    }

    @Override
    public int syevd(char jobz, char uplo, INDArray A, INDArray w) {
        return 0;
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z, int[] isuppz) {
        throw new UnsupportedOperationException();
    }

    //@Override
    public int syevr(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z, int[] isuppz) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void posv(char uplo, INDArray A, INDArray B) {
        throw new UnsupportedOperationException();

    }

    @Override
    public int geev(char jobvl, char jobvr, INDArray A, INDArray WR, INDArray WI, INDArray VL, INDArray VR) {
        throw new UnsupportedOperationException();

    }

    @Override
    public int sygvd(int itype, char jobz, char uplo, INDArray A, INDArray B, INDArray W) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void gelsd(INDArray A, INDArray B) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void geqrf(INDArray A, INDArray tau) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void ormqr(char side, char trans, INDArray A, INDArray tau, INDArray C) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void dcopy(int n, float[] dx, int dxIdx, int incx, float[] dy, int dyIdx, int incy) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void saxpy(double alpha, INDArray x, INDArray y) {
        SimpleJCublas.axpy(alpha, x, y);
    }

    /**
     * Abstraction over saxpy
     *
     * @param alpha the alpha to scale by
     * @param x     the ndarray to use
     * @param y     the ndarray to use
     */
    @Override
    public void saxpy(float alpha, INDArray x, INDArray y) {
        SimpleJCublas.saxpy(alpha, x, y);
    }


    /*
    missing functions
        gesv
        sysv
        syev
        syevx
        syevd
        syevr
        posv
        geev
        sygvd
        gelsd
        geqrf
        ormqr

     */
}
