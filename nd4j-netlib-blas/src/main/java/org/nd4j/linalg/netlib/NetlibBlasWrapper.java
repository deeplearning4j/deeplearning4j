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

package org.nd4j.linalg.netlib;

import com.github.fommil.netlib.LAPACK;
import org.jblas.ComplexDouble;
import org.jblas.ComplexFloat;
import org.jblas.NativeBlas;
import org.jblas.exceptions.SizeException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.netlib.util.intW;

import static org.jblas.util.Functions.*;


/**
 * Blas wrapper for net lib blas
 * <p/>
 * http://www.netlib.org/
 *
 * @author Adam Gibson
 */
public class NetlibBlasWrapper implements org.nd4j.linalg.factory.BlasWrapper {
    @Override
    public INDArray swap(INDArray x, INDArray y) {
        SimpleNetlibBlas.swap(x, y);
        return y;
    }

    @Override
    public INDArray scal(double alpha, INDArray x) {
        SimpleNetlibBlas.scal(alpha, x);
        return x;
    }

    @Override
    public INDArray scal(float alpha, INDArray x) {
        SimpleNetlibBlas.scal(alpha, x);
        return x;
    }


    @Override
    public IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) {
        return SimpleNetlibBlas.sscal(alpha.asFloat(), x);

    }

    @Override
    public IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) {
        return SimpleNetlibBlas.dscal(alpha, x);
    }

    @Override
    public INDArray copy(INDArray x, INDArray y) {
        SimpleNetlibBlas.copy(x, y);
        return y;
    }

    @Override
    public IComplexNDArray copy(IComplexNDArray x, IComplexNDArray y) {
        SimpleNetlibBlas.copy(x, y);
        return y;
    }

    @Override
    public INDArray axpy(double da, INDArray dx, INDArray dy) {
        SimpleNetlibBlas.axpy(da, dx, dy);
        return dy;
    }

    @Override
    public INDArray axpy(float da, INDArray dx, INDArray dy) {
        SimpleNetlibBlas.axpy(da, dx, dy);
        return dy;
    }


    @Override
    public IComplexNDArray axpy(IComplexNumber da, IComplexNDArray dx, IComplexNDArray dy) {
        SimpleNetlibBlas.axpy(da, dx, dy);
        return dy;
    }

    public double dot(INDArray x, INDArray y) {

        return SimpleNetlibBlas.dot(x, y);
    }

    //@Override
    public double dotd(INDArray x, INDArray y) {
        return SimpleNetlibBlas.dot(x, y);
    }

    @Override
    public IComplexNumber dotc(IComplexNDArray x, IComplexNDArray y) {
        return SimpleNetlibBlas.dot(x, y);
    }

    @Override
    public IComplexNumber dotu(IComplexNDArray x, IComplexNDArray y) {
        return SimpleNetlibBlas.dotu(x, y);
    }

    @Override
    public double nrm2(INDArray x) {
        return SimpleNetlibBlas.nrm2(x);
    }

    @Override
    public double nrm2(IComplexNDArray x) {
        return SimpleNetlibBlas.nrm2(x);
    }

    @Override
    public double asum(INDArray x) {
        return SimpleNetlibBlas.asum(x);

    }

    @Override
    public double asum(IComplexNDArray x) {
        return SimpleNetlibBlas.asum(x);
    }

    @Override
    public int iamax(INDArray x) {
        return SimpleNetlibBlas.iamax(x);
    }

    @Override
    public int iamax(IComplexNDArray x) {
        return SimpleNetlibBlas.iamax(x);
    }

    @Override
    public INDArray gemv(double alpha, INDArray a, INDArray x, double beta, INDArray y) {
        return SimpleNetlibBlas.gemv(a, x, y, alpha, beta);

    }

    @Override
    public INDArray gemv(float alpha, INDArray a, INDArray x, float beta, INDArray y) {
        return SimpleNetlibBlas.gemv(a, x, y, alpha, beta);
    }

    @Override
    public INDArray ger(double alpha, INDArray x, INDArray y, INDArray a) {
        return SimpleNetlibBlas.ger(x, y, a, alpha);
    }

    @Override
    public INDArray ger(float alpha, INDArray x, INDArray y, INDArray a) {
        return SimpleNetlibBlas.ger(x, y, a, alpha);

    }

    @Override
    public IComplexNDArray geru(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleNetlibBlas.geru(alpha, x, y, a);
    }


    @Override
    public IComplexNDArray geru(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleNetlibBlas.geru(alpha, x, y, a);
    }

    @Override
    public IComplexNDArray gerc(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleNetlibBlas.gerc(x, y, a, alpha.asDouble());
    }

    @Override
    public IComplexNDArray gerc(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleNetlibBlas.gerc(x, y, a, alpha.asDouble());

    }

    @Override
    public INDArray gemm(double alpha, INDArray a, INDArray b, double beta, INDArray c) {
        return SimpleNetlibBlas.gemm(a, b, c, alpha, beta);
    }

    @Override
    public INDArray gemm(float alpha, INDArray a, INDArray b, float beta, INDArray c) {
        return SimpleNetlibBlas.gemm(a, b, c, alpha, beta);
    }


    @Override
    public IComplexNDArray gemm(IComplexNumber alpha, IComplexNDArray a, IComplexNDArray b, IComplexNumber beta, IComplexNDArray c) {
        SimpleNetlibBlas.gemm(a, b, alpha, c, beta);
        return c;
    }


    @Override
    public IComplexNDArray gemv(IComplexDouble alpha, IComplexNDArray a, IComplexNDArray x, IComplexDouble beta, IComplexNDArray y) {
        DataTypeValidation.assertDouble(a, x, y);
        if (y.isScalar())
            return y.putScalar(0, dotc(a, x));
        NativeBlas.zgemv(
                'N',
                a.rows(),
                a.columns(),
                (ComplexDouble) alpha,
                a.data().asDouble(),
                a.blasOffset(),
                a.rows(),
                x.data().asDouble(),
                x.offset(),
                x.secondaryStride(),
                (ComplexDouble) beta,
                y.data().asDouble(),
                y.blasOffset(),
                y.secondaryStride()
        );
        return y;

    }

    @Override
    public IComplexNDArray gemv(IComplexFloat alpha, IComplexNDArray a, IComplexNDArray x, IComplexFloat beta, IComplexNDArray y) {
        DataTypeValidation.assertDouble(a, x, y);
        NativeBlas.cgemv(
                'N',
                a.rows(),
                a.columns(),
                (ComplexFloat) alpha,
                a.data().asFloat(),
                a.blasOffset(),
                a.rows(),
                x.data().asFloat(),
                x.offset(),
                x.secondaryStride(),
                (ComplexFloat) beta,
                y.data().asFloat(),
                y.blasOffset(),
                y.secondaryStride()
        );
        return y;
    }


    @Override
    public INDArray gesv(INDArray a, int[] ipiv, INDArray b) {
        //  public static native int sgesv(int n, int nrhs, float[] a, int aIdx, int lda, int[] ipiv, int ipivIdx, float[] b, int bIdx, int ldb);
        intW work = new intW(0);
        if (a.data().dataType() == DataBuffer.FLOAT) {
            LAPACK.getInstance().sgesv(
                    a.rows(),
                    b.columns(),
                    a.data().asFloat(),
                    a.offset(),
                    a.rows(),
                    ipiv,
                    0,
                    b.data().asFloat(),
                    b.offset(),
                    b.rows(),
                    work


            );
            return b;
        } else {
            LAPACK.getInstance().dgesv(
                    a.rows(),
                    b.columns(),
                    a.data().asDouble(),
                    a.offset(),
                    a.rows(),
                    ipiv,
                    0,
                    b.data().asDouble(),
                    b.offset(),
                    b.rows(),
                    work


            );
            return b;
        }
    }

    @Override
    public void checkInfo(String name, int info) {

    }

    @Override
    public INDArray sysv(char uplo, INDArray a, int[] ipiv, INDArray b) {
        org.netlib.util.intW info = new intW(0);
        int lwork = 0;
        if (a.data().dataType() == DataBuffer.FLOAT) {
            float[] work = new float[1];

            LAPACK.getInstance().ssysv(
                    String.valueOf(uplo),
                    a.rows(),
                    b.columns(),
                    a.data().asFloat(),
                    a.offset(),
                    a.rows(),
                    ipiv,
                    0,
                    b.data().asFloat(),
                    b.offset(),
                    b.rows(),
                    work,
                    0,
                    lwork,
                    info

            );
        } else {
            double[] work = new double[1];

            LAPACK.getInstance().dsysv(
                    String.valueOf(uplo),
                    a.rows(),
                    b.columns(),
                    a.data().asDouble(),
                    a.offset(),
                    a.rows(),
                    ipiv,
                    0,
                    b.data().asDouble(),
                    b.offset(),
                    b.rows(),
                    work,
                    0,
                    lwork,
                    info

            );
        }

        return b;
    }

    @Override
    public int syev(char jobz, char uplo, INDArray a, INDArray w) {
        intW info = new intW(0);
        int lWork = a.rows() * 5;
        if (a.data().dataType() == DataBuffer.FLOAT) {
            float[] work2 = new float[lWork];
            LAPACK.getInstance().ssyev(
                    String.valueOf(jobz),
                    String.valueOf(uplo),
                    a.rows(),
                    a.data().asFloat(),
                    a.rows(),
                    w.data().asFloat(),
                    work2,
                    lWork,
                    info
            );
        } else {
            double[] work2 = new double[lWork];
            LAPACK.getInstance().dsyev(
                    String.valueOf(jobz),
                    String.valueOf(uplo),
                    a.rows(),
                    a.data().asDouble(),
                    a.rows(),
                    w.data().asDouble(),
                    work2,
                    lWork,
                    info
            );
        }
        return info.val;
    }

    @Override
    public int syevx(char jobz,
                     char range,
                     char uplo,
                     INDArray a,
                     double vl,
                     double vu,
                     int il, int iu,
                     double abstol,
                     INDArray w,
                     INDArray z) {
        return 0;
    }

    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z) {
        return 0;
    }


    @Override
    public int syevd(char jobz, char uplo, INDArray A, INDArray w) {

        int n = A.rows();
        org.netlib.util.intW info = new intW(0);


        int lwork = 0;
        int[] iwork = new int[1];
        int liwork = 0;
        if (A.data().dataType() == DataBuffer.FLOAT) {
            float[] work = new float[1];

            LAPACK.getInstance().ssyevd(

                    String.valueOf(jobz),
                    String.valueOf(uplo),
                    n,
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    w.data().asFloat(),
                    w.offset(),
                    work,
                    0,
                    lwork,
                    iwork,
                    0,
                    liwork,
                    info);
        } else {
            double[] work = new double[1];

            LAPACK.getInstance().dsyevd(

                    String.valueOf(jobz),
                    String.valueOf(uplo),
                    n,
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    w.data().asDouble(),
                    w.offset(),
                    work,
                    0,
                    lwork,
                    iwork,
                    0,
                    liwork,
                    info);
        }


        return info.val;
    }


    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z, int[] isuppz) {
        int n = a.rows();
        org.netlib.util.intW info = new intW(0);
        org.netlib.util.intW m = new intW(0);
        double[] work = new double[1];
        int lwork = -1;
        int[] iwork = new int[1];
        int liwork = -1;


        LAPACK.getInstance().dsyevr(
                String.valueOf(jobz),
                String.valueOf(range),
                String.valueOf(uplo),
                n,
                a.data().asDouble(),
                a.offset(),
                a.rows(),
                vl,
                vu,
                il,
                iu,
                abstol,
                m,
                w.data().asDouble(),
                w.offset(),
                z.data().asDouble(),
                z.offset(),
                z.rows(),
                isuppz,
                0,
                work,
                0,
                lwork,
                iwork,
                liwork,
                0,
                info

        );


        return info.val;
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z, int[] isuppz) {
        int n = a.rows();
        org.netlib.util.intW info = new intW(0);
        org.netlib.util.intW m = new intW(0);
        float[] work = new float[1];
        int lwork = -1;
        int[] iwork = new int[1];
        int liwork = -1;


        LAPACK.getInstance().ssyevr(
                String.valueOf(jobz),
                String.valueOf(range),
                String.valueOf(uplo),
                n,
                a.data().asFloat(),
                a.offset(),
                a.rows(),
                vl,
                vu,
                il,
                iu,
                abstol,
                m,
                w.data().asFloat(),
                w.offset(),
                z.data().asFloat(),
                z.offset(),
                z.rows(),
                isuppz,
                0,
                work,
                0,
                lwork,
                iwork,
                liwork,
                0,
                info

        );


        return info.val;
    }


    @Override
    public void posv(char uplo, INDArray A, INDArray B) {
        int n = A.rows();
        int nrhs = B.columns();
        org.netlib.util.intW info = new intW(0);
        if (A.data().dataType() == DataBuffer.FLOAT)
            LAPACK.getInstance().sposv(
                    String.valueOf(uplo),
                    n,
                    nrhs,
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    B.data().asFloat(),
                    B.offset(),
                    B.rows(),
                    info

            );
        else
            LAPACK.getInstance().dposv(
                    String.valueOf(uplo),
                    n,
                    nrhs,
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    B.data().asDouble(),
                    B.offset(),
                    B.rows(),
                    info

            );

    }

    @Override
    public int geev(char jobvl, char jobvr, INDArray A, INDArray WR, INDArray WI, INDArray VL, INDArray VR) {
        intW info = new intW(0);
        int lwork = A.rows() * 5;

        if (A.data().dataType() == DataBuffer.FLOAT) {

            float[] work = new float[A.rows() * 5];

            LAPACK.getInstance().sgeev(
                    String.valueOf(jobvl),//jobvl
                    String.valueOf(jobvr),//jobvr
                    A.rows(),
                    A.data().asFloat(),//n
                    A.offset(),//a
                    A.rows(),//
                    WR.data().asFloat(),
                    WR.offset(),
                    WI.data().asFloat(),
                    WI.offset(),
                    VL.data().asFloat(),
                    VL.offset(),
                    A.rows(),
                    VR.data().asFloat(),
                    VR.offset(),
                    A.rows(),
                    work,
                    0,
                    lwork,
                    info


            );
        } else {

            double[] work = new double[A.rows() * 5];

            LAPACK.getInstance().dgeev(
                    String.valueOf(jobvl),//jobvl
                    String.valueOf(jobvr),//jobvr
                    A.rows(),
                    A.data().asDouble(),//n
                    A.offset(),//a
                    A.rows(),//
                    WR.data().asDouble(),
                    WR.offset(),
                    WI.data().asDouble(),
                    WI.offset(),
                    VL.data().asDouble(),
                    VL.offset(),
                    A.rows(),
                    VR.data().asDouble(),
                    VR.offset(),
                    A.rows(),
                    work,
                    0,
                    lwork,
                    info


            );
        }


        return info.val;
    }

    @Override
    public int sygvd(int itype, char jobz, char uplo, INDArray A, INDArray B, INDArray W) {
        int lwork = getLWork(A.rows(), jobz);
        intW info = new intW(0);
        int liwork = getLiWork(A.rows(), jobz);

        int[] iwork = new int[Math.max(1, liwork)];
        if (A.data().dataType() == DataBuffer.FLOAT) {
            float[] work = new float[lwork];

            LAPACK.getInstance().ssygvd(
                    itype,
                    String.valueOf(jobz),
                    String.valueOf(uplo),
                    A.rows(),
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    B.data().asFloat(),
                    B.offset(),
                    B.rows(),
                    W.data().asFloat(),
                    W.offset(),
                    work,
                    0,
                    lwork,
                    iwork,
                    0,
                    liwork,
                    info
            );

        } else {
            double[] work = new double[lwork];

            LAPACK.getInstance().dsygvd(
                    itype,
                    String.valueOf(jobz),
                    String.valueOf(uplo),
                    A.rows(),
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    B.data().asDouble(),
                    B.offset(),
                    B.rows(),
                    W.data().asDouble(),
                    W.offset(),
                    work,
                    0,
                    lwork,
                    iwork,
                    0,
                    liwork,
                    info
            );

        }

        return info.val;
    }

    private int getLiWork(int n, char jobz) {
        //  The dimension of the array IWORK.
        // *          If N <= 1,                LIWORK >= 1.
        //       *          If JOBZ  = 'N' and N > 1, LIWORK >= 1.
        //     *          If JOBZ  = 'V' and N > 1, LIWORK >= 3 + 5*N.
        //    *

        if (n <= 1)
            return 1;
        else if (Character.toLowerCase(jobz) == 'n' && n > 1)
            return 1;
        else if (Character.toLowerCase(jobz) == 'v' && n > 1)
            return 3 + (5 * n);
        return -1;
    }

    private int getLWork(int n, char jobz) {
        // If N <= 1,               LWORK >= 1.
        //*          If JOBZ = 'N' and N > 1, LWORK >= 2*N+1.
        //      *          If JOBZ = 'V' and N > 1, LWORK >= 1 + 6*N + 2*N**2.
        //    *
        if (n <= 1)
            return 1;
        if (Character.toLowerCase(jobz) == 'n' && n > 1)
            return 2 * n + 1;
        if (Character.toLowerCase(jobz) == 'v' && n > 1)
            return 1 + (6 * n) + 2 * (int) Math.pow(n, 2);
        return -1;

    }


    @Override
    public void gelsd(INDArray A, INDArray B) {

        int m = A.rows();
        int n = A.columns();
        int nrhs = B.columns();
        int minmn = min(m, n);
        int maxmn = max(m, n);


        if (B.rows() < maxmn) {
            throw new SizeException("Result matrix B must be padded to contain the solution matrix X!");
        }

        int smlsiz = NativeBlas.ilaenv(9, "DGELSD", "", m, n, nrhs, 0);
        int nlvl = max(0, (int) log2(minmn / (smlsiz + 1)) + 1);

        int lwork = 0;
        int[] iwork = new int[3 * minmn * nlvl + 11 * minmn];


        if (A.data().dataType() == DataBuffer.FLOAT) {
            float[] s = new float[minmn];
            float[] work = new float[1];

            intW rank = new intW(1);
            intW info = new intW(0);

            float rCond = -1f;


            LAPACK.getInstance().sgelsd(
                    m,
                    n,
                    nrhs,
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    B.data().asFloat(),
                    B.offset(),
                    B.rows(),
                    s,
                    0,
                    rCond,
                    rank,
                    work,
                    0,
                    lwork,
                    iwork,
                    0,
                    info

            );
        } else {
            double[] s = new double[minmn];
            double[] work = new double[1];

            intW rank = new intW(1);
            intW info = new intW(0);

            float rCond = -1f;


            LAPACK.getInstance().dgelsd(
                    m,
                    n,
                    nrhs,
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    B.data().asDouble(),
                    B.offset(),
                    B.rows(),
                    s,
                    0,
                    rCond,
                    rank,
                    work,
                    0,
                    lwork,
                    iwork,
                    0,
                    info

            );
        }


    }

    @Override
    public void geqrf(INDArray A, INDArray tau) {


        int lwork = 0;
        intW status = new intW(0);
        if (A.data().dataType() == DataBuffer.FLOAT) {
            float[] work = new float[1];

            LAPACK.getInstance().sgeqrf(
                    A.rows(),
                    A.columns(),
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    tau.data().asFloat(),
                    tau.offset(),
                    work,
                    0,
                    lwork,
                    status


            );


        } else {
            double[] work = new double[1];

            LAPACK.getInstance().dgeqrf(
                    A.rows(),
                    A.columns(),
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    tau.data().asDouble(),
                    tau.offset(),
                    work,
                    0,
                    lwork,
                    status


            );


        }

    }

    @Override
    public void ormqr(char side, char trans, INDArray A, INDArray tau, INDArray C) {
        int k = tau.length();
        intW status = new intW(0);
        if (A.data().dataType() == DataBuffer.FLOAT) {
            LAPACK.getInstance().sormqr(
                    String.valueOf(side),
                    String.valueOf(trans),
                    C.rows(),
                    C.columns(),
                    k,
                    A.data().asFloat(),
                    A.offset(),
                    A.data().asFloat(),
                    tau.data().asFloat(),
                    tau.offset(),
                    C.data().asFloat(),
                    C.rows(),
                    status

            );
        } else
            LAPACK.getInstance().dormqr(
                    String.valueOf(side),
                    String.valueOf(trans),
                    C.rows(),
                    C.columns(),
                    k,
                    A.data().asDouble(),
                    A.offset(),
                    A.data().asDouble(),
                    tau.data().asDouble(),
                    tau.offset(),
                    C.data().asDouble(),
                    C.rows(),
                    status

            );

    }

    @Override
    public void dcopy(int n, float[] dx, int dxIdx, int incx, float[] dy, int dyIdx, int incy) {

    }

    @Override
    public void saxpy(double alpha, INDArray x, INDArray y) {
        SimpleNetlibBlas.axpy(alpha, x, y);
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
        SimpleNetlibBlas.axpy(alpha, x, y);
    }


}
