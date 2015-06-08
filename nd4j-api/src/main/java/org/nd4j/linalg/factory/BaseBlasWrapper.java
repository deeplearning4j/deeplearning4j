package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.blas.Level2;
import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */


public abstract class BaseBlasWrapper implements BlasWrapper {
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
    public INDArray swap(INDArray x, INDArray y) {
        return null;
    }

    @Override
    public INDArray scal(double alpha, INDArray x) {
        if(x.data().dataType() == DataBuffer.Type.FLOAT)
            return scal((float) alpha,x);
        return null;
    }

    @Override
    public INDArray scal(float alpha, INDArray x) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            return scal((double) alpha,x);
        return null;
    }

    @Override
    public IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            return scal(alpha.asDouble(),x);
        return null;
    }

    @Override
    public IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) {
        if(x.data().dataType() == DataBuffer.Type.FLOAT)
            return scal(alpha.asDouble(),x);
        return null;
    }

    @Override
    public INDArray copy(INDArray x, INDArray y) {
        return null;
    }

    @Override
    public IComplexNDArray copy(IComplexNDArray x, IComplexNDArray y) {
        return null;
    }

    @Override
    public INDArray axpy(double da, INDArray dx, INDArray dy) {
        if(dx.data().dataType() == DataBuffer.Type.FLOAT)
            return axpy((float) da,dx,dy);
        return null;
    }

    @Override
    public INDArray axpy(float da, INDArray dx, INDArray dy) {
        if(dx.data().dataType() == DataBuffer.Type.DOUBLE)
            return axpy((double) da,dx,dy);

        return null;
    }

    @Override
    public IComplexNDArray axpy(IComplexNumber da, IComplexNDArray dx, IComplexNDArray dy) {
        if(dx.data().dataType() == DataBuffer.Type.DOUBLE)
            return axpy(da.asDouble(),dx,dy);
        return axpy(da.asFloat(),dx,dy);
    }

    @Override
    public double dot(INDArray x, INDArray y) {
        return 0;
    }

    @Override
    public IComplexNumber dotc(IComplexNDArray x, IComplexNDArray y) {
        return null;
    }

    @Override
    public IComplexNumber dotu(IComplexNDArray x, IComplexNDArray y) {
        return null;
    }

    @Override
    public double nrm2(INDArray x) {
        return 0;
    }

    @Override
    public double nrm2(IComplexNDArray x) {
        return 0;
    }

    @Override
    public double asum(INDArray x) {
        return 0;
    }

    @Override
    public double asum(IComplexNDArray x) {
        return 0;
    }

    @Override
    public int iamax(INDArray x) {
        return 0;
    }

    @Override
    public int iamax(IComplexNDArray x) {
        return 0;
    }

    @Override
    public INDArray gemv(double alpha, INDArray a, INDArray x, double beta, INDArray y) {
        if(a.data().dataType() == DataBuffer.Type.FLOAT) {
            return gemv((float) alpha,a,x,(float) beta,y);
        }
        return null;
    }

    @Override
    public INDArray gemv(float alpha, INDArray a, INDArray x, float beta, INDArray y) {
        if(a.data().dataType() == DataBuffer.Type.DOUBLE) {
            return gemv((double) alpha,a,x,(double) beta,y);
        }
        return null;
    }

    @Override
    public INDArray ger(double alpha, INDArray x, INDArray y, INDArray a) {
        if(x.data().dataType() == DataBuffer.Type.FLOAT) {
            return ger((float) alpha,x,y,a);
        }
        return null;
    }

    @Override
    public INDArray ger(float alpha, INDArray x, INDArray y, INDArray a) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE) {
            return ger((double) alpha,x,y,a);
        }
        return null;
    }

    @Override
    public IComplexNDArray gemv(IComplexDouble alpha, IComplexNDArray a, IComplexNDArray x, IComplexDouble beta, IComplexNDArray y) {
        if(x.data().dataType() == DataBuffer.Type.FLOAT) {
            return gemv(alpha.asFloat(),a,x,beta.asFloat(),y);
        }
        return null;
    }

    @Override
    public IComplexNDArray gemv(IComplexFloat alpha, IComplexNDArray a, IComplexNDArray x, IComplexFloat beta, IComplexNDArray y) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE) {
            return gemv(alpha.asDouble(),a,x,beta.asDouble(),y);
        }
        return null;
    }

    @Override
    public IComplexNDArray geru(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        if(x.data().dataType() == DataBuffer.Type.FLOAT) {
            return geru(alpha.asFloat(), x, y, a);
        }
        return null;
    }

    @Override
    public IComplexNDArray geru(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE) {
            return geru(alpha.asDouble(), x, y, a);
        }
        return null;
    }

    @Override
    public IComplexNDArray gerc(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE) {
            return gerc(alpha.asDouble(), x, y, a);
        }
        return null;
    }

    @Override
    public IComplexNDArray gerc(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        if(x.data().dataType() == DataBuffer.Type.FLOAT) {
            return gerc(alpha.asFloat(), x, y, a);
        }
        return null;
    }

    @Override
    public INDArray gemm(double alpha, INDArray a, INDArray b, double beta, INDArray c) {
        if(a.data().dataType() == DataBuffer.Type.FLOAT) {
            return gemm((float) alpha,a,b,(float) beta,c);
        }
        return null;
    }

    @Override
    public INDArray gemm(float alpha, INDArray a, INDArray b, float beta, INDArray c) {
        if(a.data().dataType() == DataBuffer.Type.DOUBLE) {
            return gemm((double) alpha,a,b,(double) beta,c);
        }
        return null;
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
    public int syevx(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z) {
        return 0;
    }

    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z) {
        return 0;
    }

    @Override
    public int syevd(char jobz, char uplo, INDArray A, INDArray w) {
        return 0;
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z, int[] isuppz) {
        return 0;
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z, int[] isuppz) {
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
    public void saxpy(double alpha, INDArray x, INDArray y) {
        if(x.data().dataType() == DataBuffer.Type.FLOAT) {
            saxpy((float) alpha,x,y);
        }
        else {

        }
    }

    @Override
    public void saxpy(float alpha, INDArray x, INDArray y) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE) {
            saxpy((double) alpha,x,y);
        }
        else {

        }
    }


    

}
