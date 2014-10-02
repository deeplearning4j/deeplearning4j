package org.nd4j.linalg.netlib;

import com.github.fommil.netlib.LAPACK;
import org.jblas.NativeBlas;
import org.jblas.exceptions.SizeException;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.netlib.util.intW;

import static org.jblas.util.Functions.log2;
import static org.jblas.util.Functions.max;
import static org.jblas.util.Functions.min;


/**
 * Blas wrapper for net lib blas
 *
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
    public INDArray scal(float alpha, INDArray x) {
        SimpleNetlibBlas.scal(alpha, x);
        return x;
    }



    @Override
    public IComplexNDArray scal(IComplexNumber alpha, IComplexNDArray x) {
        return SimpleNetlibBlas.sscal(alpha.asFloat(), x);

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
    public INDArray axpy(float da, INDArray dx, INDArray dy) {
        SimpleNetlibBlas.axpy(da, dx, dy);
        return dy;
    }



    @Override
    public IComplexNDArray axpy(IComplexNumber da, IComplexNDArray dx, IComplexNDArray dy) {
        SimpleNetlibBlas.axpy(da, dx, dy);
        return dy;
    }

    public float dot(INDArray x, INDArray y) {

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
    public INDArray gemv(float alpha, INDArray a, INDArray x, float beta, INDArray y) {
        return SimpleNetlibBlas.gemv(a, x, y, alpha, beta);
    }

    @Override
    public INDArray ger(float alpha, INDArray x, INDArray y, INDArray a) {
        return null;
    }


    @Override
    public IComplexNDArray geru(IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleNetlibBlas.geru(alpha,x,y,a);
    }

    @Override
    public IComplexNDArray gerc(IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleNetlibBlas.gerc(x, y, a, alpha.asDouble());
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
    public INDArray gesv(INDArray a, int[] ipiv, INDArray b) {

        return null;
    }

    @Override
    public void checkInfo(String name, int info) {

    }

    @Override
    public INDArray sysv(char uplo, INDArray a, int[] ipiv, INDArray b) {
        org.netlib.util.intW info = new intW(0);
        float[] work = new float[1];
        int lwork = 0;
        LAPACK.getInstance().ssysv(
                String.valueOf(uplo),
                a.rows(),
                b.columns(),
                a.data(),
                a.offset(),
                a.rows(),
                ipiv,
                0,
                b.data(),
                b.offset(),
                b.rows(),
                work,
                0,
                lwork,
                info

        );

        return b;
    }

    @Override
    public int syev(char jobz, char uplo, INDArray a, INDArray w) {
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


        float[] work = new float[1];
        int lwork = 0;
        int[] iwork = new int[1];
        int liwork = 0;
        LAPACK.getInstance().ssyevd(

                String.valueOf(jobz),
                String.valueOf(uplo),
                n,
                A.data(),
                A.offset(),
                A.rows(),
                w.data(),
                w.offset(),
                work,
                0,
                lwork,
                iwork,
                0,
                liwork,
                info );



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
                a.data(),
                a.offset(),
                a.rows(),
                vl,
                vu,
                il,
                iu,
                abstol,
                m,
                w.data(),
                w.offset(),
                z.data(),
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
        LAPACK.getInstance().sposv(
                String.valueOf(uplo),
                n,
                nrhs,
                A.data(),
                A.offset(),
                A.rows(),
                B.data(),
                B.offset(),
                B.rows(),
                info

        );

    }

    @Override
    public int geev(char jobvl, char jobvr, INDArray A, INDArray WR, INDArray WI, INDArray VL, INDArray VR) {
        float[] work = new float[A.rows() * 5];
        intW info = new intW(0);
        int lwork = A.rows() * 5;

        LAPACK.getInstance().sgeev(
                String.valueOf(jobvl),//jobvl
                String.valueOf(jobvr),//jobvr
                A.rows(),
                A.data(),//n
                A.offset(),//a
                A.rows(),//
                WR.data(),
                WR.offset(),
                WI.data(),
                WI.offset(),
                VL.data(),
                VL.offset(),
                A.rows(),
                VR.data(),
                VR.offset(),
                A.rows(),
                work,
                0,
                lwork,
                info


        );

        return info.val;
    }

    @Override
    public int sygvd(int itype, char jobz, char uplo, INDArray A, INDArray B, INDArray W) {
        int lwork = getLWork(A.rows(),jobz);
        intW info = new intW(0);
        float[] work = new float[lwork];
        int liwork = getLiWork(A.rows(),jobz);

        int[] iwork = new int[Math.max(1,liwork)];


        LAPACK.getInstance().ssygvd(
                itype,
                String.valueOf(jobz),
                String.valueOf(uplo),
                A.rows(),
                A.data(),
                A.offset(),
                A.rows(),
                B.data(),
                B.offset(),
                B.rows(),
                W.data(),
                W.offset(),
                work,
                0,
                lwork,
                iwork,
                0,
                liwork,
                info
        );

        return info.val;
    }

    private int getLiWork(int n,char jobz) {
        //  The dimension of the array IWORK.
        // *          If N <= 1,                LIWORK >= 1.
        //       *          If JOBZ  = 'N' and N > 1, LIWORK >= 1.
        //     *          If JOBZ  = 'V' and N > 1, LIWORK >= 3 + 5*N.
        //    *

        if(n <= 1)
            return 1;
        else if(Character.toLowerCase(jobz) == 'n' && n > 1)
            return 1;
        else if(Character.toLowerCase(jobz) == 'v' && n > 1)
            return 3 + (5 * n);
        return -1;
    }

    private int getLWork(int n,char jobz) {
        // If N <= 1,               LWORK >= 1.
        //*          If JOBZ = 'N' and N > 1, LWORK >= 2*N+1.
        //      *          If JOBZ = 'V' and N > 1, LWORK >= 1 + 6*N + 2*N**2.
        //    *
        if(n <= 1)
            return 1;
        if(Character.toLowerCase(jobz) == 'n' && n > 1)
            return 2 * n + 1;
        if(Character.toLowerCase(jobz) == 'v' && n > 1)
            return 1 + (6 * n) + 2 * (int) Math.pow(n,2);
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
        int nlvl = max(0, (int) log2(minmn/ (smlsiz+1)) + 1);

        int lwork = 0;
        int[] iwork = new int[3 * minmn * nlvl + 11 * minmn];
        float[] s = new float[minmn];
        float[] work = new float[1];

        intW rank = new intW(1);
        intW info = new intW(0);

        float rCond = -1f;


        LAPACK.getInstance().sgelsd(
                m,
                n,
                nrhs,
                A.data(),
                A.offset(),
                A.rows(),
                B.data(),
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

    @Override
    public void geqrf(INDArray A, INDArray tau) {


        float[] work = new float[1];
        int lwork = 0;
        intW status = new intW(0);
        LAPACK.getInstance().sgeqrf(
                A.rows(),
                A.columns(),
                A.data(),
                A.offset(),
                A.rows(),
                tau.data(),
                tau.offset(),
                work,
                0,
                lwork,
                status


        );



    }

    @Override
    public void ormqr(char side, char trans, INDArray A, INDArray tau, INDArray C) {
        int k = tau.length();
        intW status = new intW(0);

        LAPACK.getInstance().sormqr(
                String.valueOf(side),
                String.valueOf(trans),
                C.rows(),
                C.columns(),
                k,
                A.data(),
                A.offset(),
                A.data(),
                tau.data(),
                tau.offset(),
                C.data(),
                C.rows(),
                status

        );
    }

    @Override
    public void dcopy(int n, float[] dx, int dxIdx, int incx, float[] dy, int dyIdx, int incy) {

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
        SimpleNetlibBlas.saxpy(alpha, x, y);
    }



}
