package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.blas.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.LinAlgExceptions;

/**
 *
 * Base implementation of a blas wrapper that
 * delegates things to the underlying level.
 * This is a migration tool to preserve the old api
 * while allowing for migration to the newer one.
 *
 * @author Adam Gibson
 */

public abstract class BaseBlasWrapper implements BlasWrapper {

    @Override
    public Lapack lapack() {
        return Nd4j.factory().lapack();
    }

    @Override
    public Level1 level1() {
        return Nd4j.factory().level1();
    }

    @Override
    public Level2 level2() {
        return Nd4j.factory().level2();

    }

    @Override
    public Level3 level3() {
        return Nd4j.factory().level3();

    }

    @Override
    public INDArray axpy(Number da, INDArray dx, INDArray dy) {
        if(!dx.isVector())
            throw new IllegalArgumentException("Unable to use axpy on a non vector");
        LinAlgExceptions.assertSameShape(dx,dy);
        level1().axpy(dx.length(),da.doubleValue(),dx,dy);
        return dy;
    }

    @Override
    public IComplexNDArray scal(IComplexNumber alpha, IComplexNDArray x) {
        LinAlgExceptions.assertVector(x);
        level1().scal(x.length(),alpha,x);
        return x;
    }

    @Override
    public INDArray gemv(Number alpha, INDArray a, INDArray x, double beta, INDArray y) {
        LinAlgExceptions.assertVector(x, y);
        LinAlgExceptions.assertMatrix(a);
        level2().gemv(BlasBufferUtil.getCharForTranspose(a),BlasBufferUtil.getCharForTranspose(x),alpha.doubleValue(),a,x,beta,y);
        return y;
    }

    @Override
    public INDArray ger(Number alpha, INDArray x, INDArray y, INDArray a) {
        level2().ger(BlasBufferUtil.getCharForTranspose(x),alpha.doubleValue(),x,y,a);
        return a;
    }

    @Override
    public IComplexNDArray gemv(IComplexNumber alpha, IComplexNDArray a, IComplexNDArray x, IComplexNumber beta, IComplexNDArray y) {
        LinAlgExceptions.assertVector(x,y);
        LinAlgExceptions.assertMatrix(a);
        level2().gemv('N','N',alpha,a,x,beta,y);
        return y;
    }

    @Override
    public IComplexNDArray geru(IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        level2().geru('N',alpha,x,y,a);
        return a;
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, Number abstol, INDArray w, INDArray z, int[] isuppz) {
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray swap(INDArray x, INDArray y) {
        level1().swap(x,y);
        return y;
    }

    @Override
    public INDArray scal(double alpha, INDArray x) {
        LinAlgExceptions.assertVector(x);

        if(x.data().dataType() == DataBuffer.Type.FLOAT)
            return scal((float) alpha,x);
        level1().scal(x.length(),alpha,x);
        return x;
    }

    @Override
    public INDArray scal(float alpha, INDArray x) {
        LinAlgExceptions.assertVector(x);

        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            return scal((double) alpha,x);
        level1().scal(x.length(),alpha,x);
        return x;
    }

    @Override
    public IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) {
        LinAlgExceptions.assertVector(x);

        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            return scal(alpha.asDouble(),x);
        level1().scal(x.length(),alpha,x);
        return x;
    }

    @Override
    public IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) {
        LinAlgExceptions.assertVector(x);

        if(x.data().dataType() == DataBuffer.Type.FLOAT)
            return scal(alpha.asDouble(),x);
        level1().scal(x.length(),alpha,x);
        return x;
    }

    @Override
    public INDArray copy(INDArray x, INDArray y) {
        LinAlgExceptions.assertVector(x,y);
        level1().copy(x,y);
        return y;
    }

    @Override
    public IComplexNDArray copy(IComplexNDArray x, IComplexNDArray y) {
        LinAlgExceptions.assertVector(x,y);
        level1().copy(x,y);
        return y;
    }

    @Override
    public INDArray axpy(double da, INDArray dx, INDArray dy) {
        LinAlgExceptions.assertVector(dx,dy);

        if(dx.data().dataType() == DataBuffer.Type.FLOAT)
            return axpy((float) da,dx,dy);
        level1().axpy(dx.length(),da,dx,dy);
        return dy;
    }

    @Override
    public INDArray axpy(float da, INDArray dx, INDArray dy) {
        LinAlgExceptions.assertVector(dx,dy);

        if(dx.data().dataType() == DataBuffer.Type.DOUBLE)
            return axpy((double) da,dx,dy);

        level1().axpy(dx.length(),da,dx,dy);
        return dy;
    }

    @Override
    public IComplexNDArray axpy(IComplexNumber da, IComplexNDArray dx, IComplexNDArray dy) {
        LinAlgExceptions.assertVector(dx,dy);

        if(dx.data().dataType() == DataBuffer.Type.DOUBLE)
            return axpy(da.asDouble(),dx,dy);
        return axpy(da.asFloat(),dx,dy);
    }

    @Override
    public double dot(INDArray x, INDArray y) {
        return level1().dot(x.length(),1,x,y);
    }

    @Override
    public IComplexNumber dotc(IComplexNDArray x, IComplexNDArray y) {
        LinAlgExceptions.assertVector(x,y);
        return level1().dot(x.length(),Nd4j.UNIT,x,y);
    }

    @Override
    public IComplexNumber dotu(IComplexNDArray x, IComplexNDArray y) {
        LinAlgExceptions.assertVector(x,y);
        return level1().dot(x.length(),Nd4j.UNIT,x,y);
    }

    @Override
    public double nrm2(INDArray x) {
        LinAlgExceptions.assertVector(x);
        return level1().nrm2(x);
    }

    @Override
    public IComplexNumber nrm2(IComplexNDArray x) {
        LinAlgExceptions.assertVector(x);
        return level1().nrm2(x);

    }

    @Override
    public double asum(INDArray x) {
        LinAlgExceptions.assertVector(x);
        return level1().asum(x);
    }

    @Override
    public IComplexNumber asum(IComplexNDArray x) {
        LinAlgExceptions.assertVector(x);
        return level1().asum(x);

    }

    @Override
    public int iamax(INDArray x) {
        return level1().iamax(x);
    }

    @Override
    public int iamax(IComplexNDArray x) {
        LinAlgExceptions.assertVector(x);
        return level1().iamax(x);
    }

    @Override
    public INDArray gemv(double alpha, INDArray a, INDArray x, double beta, INDArray y) {
        LinAlgExceptions.assertVector(x,y);
        LinAlgExceptions.assertMatrix(a);

        if(a.data().dataType() == DataBuffer.Type.FLOAT) {
            return gemv((float) alpha,a,x,(float) beta,y);
        }
        level2().gemv('N','N',alpha,a,x,beta,y);
        return y;
    }

    @Override
    public INDArray gemv(float alpha, INDArray a, INDArray x, float beta, INDArray y) {
        LinAlgExceptions.assertVector(x,y);
        LinAlgExceptions.assertMatrix(a);

        if(a.data().dataType() == DataBuffer.Type.DOUBLE) {
            return gemv((double) alpha,a,x,(double) beta,y);
        }
        level2().gemv('N','N',alpha,a,x,beta,y);
        return y;
    }

    @Override
    public INDArray ger(double alpha, INDArray x, INDArray y, INDArray a) {
        LinAlgExceptions.assertVector(x,y);
        LinAlgExceptions.assertMatrix(a);

        if(x.data().dataType() == DataBuffer.Type.FLOAT) {
            return ger((float) alpha,x,y,a);
        }

        level2().ger('N',alpha,x,y,a);
        return a;
    }

    @Override
    public INDArray ger(float alpha, INDArray x, INDArray y, INDArray a) {
        LinAlgExceptions.assertVector(x,y);
        LinAlgExceptions.assertMatrix(a);


        if(x.data().dataType() == DataBuffer.Type.DOUBLE) {
            return ger((double) alpha,x,y,a);
        }

        level2().ger('N',alpha,x,y,a);
        return a;
    }

    @Override
    public IComplexNDArray gemv(IComplexDouble alpha, IComplexNDArray a, IComplexNDArray x, IComplexDouble beta, IComplexNDArray y) {
        LinAlgExceptions.assertVector(x,y);
        LinAlgExceptions.assertMatrix(a);

        if(x.data().dataType() == DataBuffer.Type.FLOAT) {
            return gemv(alpha.asFloat(),a,x,beta.asFloat(),y);
        }

        level2().gemv('N','N',alpha,a,x,beta,y);
        return y;
    }

    @Override
    public IComplexNDArray gemv(IComplexFloat alpha, IComplexNDArray a, IComplexNDArray x, IComplexFloat beta, IComplexNDArray y) {
        LinAlgExceptions.assertVector(x,y);
        LinAlgExceptions.assertMatrix(a);


        if(x.data().dataType() == DataBuffer.Type.DOUBLE) {
            return gemv(alpha.asDouble(),a,x,beta.asDouble(),y);
        }

        level2().gemv('N','N',alpha,a,x,beta,y);
        return y;
    }

    @Override
    public IComplexNDArray geru(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        LinAlgExceptions.assertVector(x,y);
        LinAlgExceptions.assertMatrix(a);

        if(x.data().dataType() == DataBuffer.Type.FLOAT) {
            return geru(alpha.asFloat(), x, y, a);
        }
        level2().geru('N',alpha,x,y,a);
        return a;
    }

    @Override
    public IComplexNDArray geru(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        LinAlgExceptions.assertVector(x,y);
        LinAlgExceptions.assertMatrix(a);

        if(x.data().dataType() == DataBuffer.Type.DOUBLE) {
            return geru(alpha.asDouble(), x, y, a);
        }
        level2().geru('N',alpha,x,y,a);
        return a;
    }

    @Override
    public IComplexNDArray gerc(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE) {
            return gerc(alpha.asDouble(), x, y, a);
        }
        gerc(alpha,x,y,a);
        return a;
    }

    @Override
    public IComplexNDArray gerc(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        if(x.data().dataType() == DataBuffer.Type.FLOAT) {
            return gerc(alpha.asFloat(), x, y, a);
        }
        gerc(alpha,x,y,a);
        return a;
    }

    @Override
    public INDArray gemm(double alpha, INDArray a, INDArray b, double beta, INDArray c) {
        LinAlgExceptions.assertMatrix(a,b,c);

        if(a.data().dataType() == DataBuffer.Type.FLOAT) {
            return gemm((float) alpha,a,b,(float) beta,c);
        }


        level3().gemm(
                BlasBufferUtil.getCharForTranspose(a),
                BlasBufferUtil.getCharForTranspose(b)
                ,BlasBufferUtil.getCharForTranspose(c)
                ,alpha
                ,a,b
                ,beta
                ,c);
        return c;
    }

    @Override
    public INDArray gemm(float alpha, INDArray a, INDArray b, float beta, INDArray c) {
        LinAlgExceptions.assertMatrix(a,b,c);


        if(a.data().dataType() == DataBuffer.Type.DOUBLE) {
            return gemm((double) alpha,a,b,(double) beta,c);
        }

        level3().gemm(
                BlasBufferUtil.getCharForTranspose(a),
                BlasBufferUtil.getCharForTranspose(b)
                ,BlasBufferUtil.getCharForTranspose(c)
                ,alpha
                ,a
                ,b
                ,beta
                ,c);
        return c;
    }

    @Override
    public IComplexNDArray gemm(IComplexNumber alpha, IComplexNDArray a, IComplexNDArray b, IComplexNumber beta, IComplexNDArray c) {
        LinAlgExceptions.assertMatrix(a,b,c);

        level3().gemm(
                BlasBufferUtil.getCharForTranspose(a),
                BlasBufferUtil.getCharForTranspose(b)
                ,BlasBufferUtil.getCharForTranspose(c)
                ,alpha
                ,a
                ,b
                ,beta
                ,c);
        return c;
    }

    @Override
    public INDArray gesv(INDArray a, int[] ipiv, INDArray b) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void checkInfo(String name, int info) {

    }

    @Override
    public INDArray sysv(char uplo, INDArray a, int[] ipiv, INDArray b) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int syev(char jobz, char uplo, INDArray a, INDArray w) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int syevd(char jobz, char uplo, INDArray A, INDArray w) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z, int[] isuppz) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z, int[] isuppz) {
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
    public void saxpy(double alpha, INDArray x, INDArray y) {
        axpy(alpha,x,y);

    }

    @Override
    public void saxpy(float alpha, INDArray x, INDArray y) {
        axpy(alpha,x,y);
    }




}
