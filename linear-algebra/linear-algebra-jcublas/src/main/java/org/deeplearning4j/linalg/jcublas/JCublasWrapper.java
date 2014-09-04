package org.deeplearning4j.linalg.jcublas;

import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.ndarray.INDArray;


/**
 * Created by mjk on 8/20/14.
 */
public class JCublasWrapper implements org.deeplearning4j.linalg.factory.BlasWrapper {
    @Override
    public INDArray swap(INDArray x, INDArray y) {
        SimpleJCublas.swap(x,y);
        return y;
    }

    @Override
    public INDArray scal(float alpha, INDArray x) {
        SimpleJCublas.scal(alpha,x);
        return x;
    }



    @Override
    public IComplexNDArray scal(IComplexNumber alpha, IComplexNDArray x) {
        return SimpleJCublas.zscal(alpha.asDouble(), x);

    }

    @Override
    public INDArray copy(INDArray x, INDArray y) {
        SimpleJCublas.copy(x,y);
        return y;
    }

    @Override
    public IComplexNDArray copy(IComplexNDArray x, IComplexNDArray y) {
        SimpleJCublas.copy(x,y);
        return y;
    }

    @Override
    public INDArray axpy(float da, INDArray dx, INDArray dy) {
        SimpleJCublas.axpy(da,dx,dy);
        return dy;
    }



    @Override
    public IComplexNDArray axpy(IComplexNumber da, IComplexNDArray dx, IComplexNDArray dy) {
        SimpleJCublas.axpy(da,dx,dy);
        return dy;
    }

    public float dot(INDArray x, INDArray y) {
        return 0.0f;
    }
    //@Override
    public double dotd(INDArray x, INDArray y) {
        return SimpleJCublas.dot(x,y);
    }

    @Override
    public IComplexNumber dotc(IComplexNDArray x, IComplexNDArray y) {
        return SimpleJCublas.dot(x,y);
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
    public INDArray gemv(float alpha, INDArray a, INDArray x, float beta, INDArray y) {
        return SimpleJCublas.gemv(a,x,y, alpha, beta);
    }

    @Override
    public INDArray ger(float alpha, INDArray x, INDArray y, INDArray a) {
        return null;
    }


    @Override
    public IComplexNDArray geru(IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleJCublas.geru(x, y, a, alpha.asDouble());
    }

    @Override
    public IComplexNDArray gerc(IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return SimpleJCublas.gerc(x, y, a, alpha.asDouble());
    }

    @Override
    public INDArray gemm(float alpha, INDArray a, INDArray b, float beta, INDArray c) {
        return SimpleJCublas.gemm(a,b,c,alpha,beta);
    }



    @Override
    public IComplexNDArray gemm(IComplexNumber alpha, IComplexNDArray a, IComplexNDArray b, IComplexNumber beta, IComplexNDArray c) {
        return null;
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
        return null;
    }

    @Override
    public int syev(char jobz, char uplo, INDArray a, INDArray w) {
        return 0;
    }

    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z) {
        return 0;
    }

    //@Override
    public int syevx(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z) {
        return 0;
    }

    @Override
    public int syevd(char jobz, char uplo, INDArray A, INDArray w) {
        return 0;
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z, int[] isuppz) {
        return 0;
    }

    //@Override
    public int syevr(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z, int[] isuppz) {
        return 0;
    }

    @Override
    public void posv(char uplo, INDArray A, INDArray B) {

    }

    @Override
    public int geev(char jobvl, char jobvr, INDArray A, INDArray WR, INDArray WI, INDArray VL, INDArray VR) {
        return 0;
    }

    @Override
    public int sygvd(int itype, char jobz, char uplo, INDArray A, INDArray B, INDArray W) {
        return 0;
    }

    @Override
    public void gelsd(INDArray A, INDArray B) {

    }

    @Override
    public void geqrf(INDArray A, INDArray tau) {

    }

    @Override
    public void ormqr(char side, char trans, INDArray A, INDArray tau, INDArray C) {

    }

    @Override
    public void dcopy(int n, float[] dx, int dxIdx, int incx, float[] dy, int dyIdx, int incy) {

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
