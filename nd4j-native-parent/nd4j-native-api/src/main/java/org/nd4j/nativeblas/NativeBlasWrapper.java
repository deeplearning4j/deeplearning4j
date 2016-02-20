package org.nd4j.nativeblas;


import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.BaseBlasWrapper;

/**
 * Wraps ndarray calls
 * to blas and contains a pointer converter
 * which converts ndarrays to their raw native addresses.
 *
 */
public class NativeBlasWrapper extends BaseBlasWrapper {
    private NativeBlas nativeBlas = new NativeBlas();
    private PointerConverter pointerConverter;
    
    @Override
    public IComplexNumber asum(IComplexNDArray x) {
        return nativeBlas.asumComplex(pointerConverter.toPointer(x));
    }

    @Override
    public double asum(INDArray x) {
        return nativeBlas.asum(pointerConverter.toPointer(x));
    }

    @Override
    public INDArray axpy(double da, INDArray dx, INDArray dy) {
        return nativeBlas.axpy(da, pointerConverter.toPointer(dx),pointerConverter.toPointer(dy));
    }

    @Override
    public INDArray axpy(float da, INDArray dx, INDArray dy) {
        return nativeBlas.axpy(da, pointerConverter.toPointer(dx),pointerConverter.toPointer(dy));
    }

    @Override
    public IComplexNDArray axpy(IComplexNumber da, IComplexNDArray dx, IComplexNDArray dy) {
        return nativeBlas.axpy(da, pointerConverter.toPointer(dx),pointerConverter.toPointer(dy));
    }

    @Override
    public INDArray axpy(Number da, INDArray dx, INDArray dy) {
        return nativeBlas.axpy(da, pointerConverter.toPointer(dx),pointerConverter.toPointer(dy));
    }

    @Override
    public void checkInfo(String name, int info) {
    }

    @Override
    public IComplexNDArray copy(IComplexNDArray x, IComplexNDArray y) {
        return nativeBlas.copyComplex(pointerConverter.toPointer(x), pointerConverter.toPointer(y));
    }

    @Override
    public INDArray copy(INDArray x, INDArray y) {
        return nativeBlas.copy(pointerConverter.toPointer(x),pointerConverter.toPointer(y));
    }

    @Override
    public double dot(INDArray x, INDArray y) {
        return nativeBlas.dot(pointerConverter.toPointer(x),pointerConverter.toPointer(y));
    }

    @Override
    public IComplexNumber dotc(IComplexNDArray x, IComplexNDArray y) {
        return nativeBlas.dotc(pointerConverter.toPointer(x), pointerConverter.toPointer(y));
    }

    @Override
    public IComplexNumber dotu(IComplexNDArray x, IComplexNDArray y) {
        return nativeBlas.dotu(pointerConverter.toPointer(x), pointerConverter.toPointer(y));
    }

    @Override
    public int geev(char jobvl, char jobvr, INDArray A, INDArray WR, INDArray WI, INDArray VL, INDArray VR) {
        return nativeBlas.geev(jobvl, jobvr,pointerConverter.toPointer(A),pointerConverter.toPointer(WR),pointerConverter.toPointer(WI),pointerConverter.toPointer(VL),pointerConverter.toPointer(VR));
    }

    @Override
    public void gelsd(INDArray A, INDArray B) {
        nativeBlas.gelsd(pointerConverter.toPointer(A),pointerConverter.toPointer(B));
    }

    @Override
    public INDArray gemm(double alpha, INDArray a, INDArray b, double beta, INDArray c) {
        return nativeBlas.gemm(alpha, pointerConverter.toPointer(a),pointerConverter.toPointer(b), beta, pointerConverter.toPointer(c));
    }

    @Override
    public INDArray gemm(float alpha, INDArray a, INDArray b, float beta, INDArray c) {
        return nativeBlas.gemm(alpha, pointerConverter.toPointer(a),pointerConverter.toPointer(b), beta, pointerConverter.toPointer(c));
    }

    @Override
    public IComplexNDArray gemm(IComplexNumber alpha, IComplexNDArray a, IComplexNDArray b, IComplexNumber beta, IComplexNDArray c) {
        return nativeBlas.gemm(alpha, pointerConverter.toPointer(a),pointerConverter.toPointer(b), beta, pointerConverter.toPointer(c));
    }

    @Override
    public INDArray gemv(double alpha, INDArray a, INDArray x, double beta, INDArray y) {
        return nativeBlas.gemv(alpha, pointerConverter.toPointer(a), pointerConverter.toPointer(x), beta, pointerConverter.toPointer(y));
    }

    @Override
    public INDArray gemv(float alpha, INDArray a, INDArray x, float beta, INDArray y) {
        return nativeBlas.gemv(alpha, pointerConverter.toPointer(a), pointerConverter.toPointer(x), beta, pointerConverter.toPointer(y));
    }

    @Override
    public IComplexNDArray gemv(IComplexDouble alpha, IComplexNDArray a, IComplexNDArray x, IComplexDouble beta, IComplexNDArray y) {
        return nativeBlas.gemv(alpha, pointerConverter.toPointer(a), pointerConverter.toPointer(x), beta, pointerConverter.toPointer(y));
    }

    @Override
    public IComplexNDArray gemv(IComplexFloat alpha, IComplexNDArray a, IComplexNDArray x, IComplexFloat beta, IComplexNDArray y) {
        return nativeBlas.gemv(alpha, pointerConverter.toPointer(a), pointerConverter.toPointer(x), beta, pointerConverter.toPointer(y));
    }

    @Override
    public IComplexNDArray gemv(IComplexNumber alpha, IComplexNDArray a, IComplexNDArray x, IComplexNumber beta, IComplexNDArray y) {
        return nativeBlas.gemv(alpha, pointerConverter.toPointer(a), pointerConverter.toPointer(x), beta, pointerConverter.toPointer(y));
    }

    @Override
    public INDArray gemv(Number alpha, INDArray a, INDArray x, double beta, INDArray y) {
        return nativeBlas.gemv(alpha, pointerConverter.toPointer(a), pointerConverter.toPointer(x), beta, pointerConverter.toPointer(y));
    }

    @Override
    public void geqrf(INDArray A, INDArray tau) {
       nativeBlas.geqrf(pointerConverter.toPointer(A), pointerConverter.toPointer(tau));
    }

    @Override
    public INDArray ger(double alpha, INDArray x, INDArray y, INDArray a) {
        return nativeBlas.ger(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y),pointerConverter.toPointer(a));
    }

    @Override
    public INDArray ger(float alpha, INDArray x, INDArray y, INDArray a) {
        return nativeBlas.ger(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y),pointerConverter.toPointer(a));
    }

    @Override
    public INDArray ger(Number alpha, INDArray x, INDArray y, INDArray a) {
        return nativeBlas.ger(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y),pointerConverter.toPointer(a));
    }

    @Override
    public IComplexNDArray gerc(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return nativeBlas.gerc(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y), pointerConverter.toPointer(a));
    }

    @Override
    public IComplexNDArray gerc(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return nativeBlas.gerc(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y), pointerConverter.toPointer(a));
    }

    @Override
    public IComplexNDArray geru(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return nativeBlas.geru(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y), pointerConverter.toPointer(a));
    }

    @Override
    public IComplexNDArray geru(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return nativeBlas.geru(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y), pointerConverter.toPointer(a));
    }

    @Override
    public IComplexNDArray geru(IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        return nativeBlas.geru(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y), pointerConverter.toPointer(a));
    }

    @Override
    public INDArray gesv(INDArray a, int[] ipiv, INDArray b) {
        return nativeBlas.gesv(pointerConverter.toPointer(a), ipiv, pointerConverter.toPointer(b));
    }

    @Override
    public int iamax(IComplexNDArray x) {
        return nativeBlas.iamax(pointerConverter.toPointer(x));
    }

    @Override
    public int iamax(INDArray x) {
        return nativeBlas.iamax(pointerConverter.toPointer(x));
    }



    @Override
    public IComplexNumber nrm2(IComplexNDArray x) {
        return nativeBlas.nrm2Complex(pointerConverter.toPointer(x));
    }

    @Override
    public double nrm2(INDArray x) {
        return nativeBlas.nrm2(pointerConverter.toPointer(x));
    }

    @Override
    public void ormqr(char side, char trans, INDArray A, INDArray tau, INDArray C) {
        nativeBlas.ormqr(side, trans, pointerConverter.toPointer(A), pointerConverter.toPointer(tau), pointerConverter.toPointer(C));
    }

    @Override
    public void posv(char uplo, INDArray A, INDArray B) {
        nativeBlas.posv(uplo, pointerConverter.toPointer(A), pointerConverter.toPointer(B));
    }

    @Override
    public void saxpy(double alpha, INDArray x, INDArray y) {
        nativeBlas.saxpy(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y));
    }

    @Override
    public void saxpy(float alpha, INDArray x, INDArray y) {
        nativeBlas.saxpy(alpha, pointerConverter.toPointer(x), pointerConverter.toPointer(y));
    }

    @Override
    public INDArray scal(double alpha, INDArray x) {
        return nativeBlas.scal(alpha, pointerConverter.toPointer(x));
    }

    @Override
    public INDArray scal(float alpha, INDArray x) {
        return nativeBlas.scal(alpha, pointerConverter.toPointer(x));
    }

    @Override
    public IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) {
        return nativeBlas.scal(alpha, pointerConverter.toPointer(x));
    }

    @Override
    public IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) {
        return nativeBlas.scal(alpha, pointerConverter.toPointer(x));
    }

    @Override
    public IComplexNDArray scal(IComplexNumber alpha, IComplexNDArray x) {
        return nativeBlas.scal(alpha, pointerConverter.toPointer(x));
    }

    @Override
    public INDArray swap(INDArray x, INDArray y) {
        return nativeBlas.swap(pointerConverter.toPointer(x),pointerConverter.toPointer(y));
    }

    @Override
    public int syev(char jobz, char uplo, INDArray a, INDArray w) {
        return nativeBlas.syev(jobz, uplo,pointerConverter.toPointer(a),pointerConverter.toPointer(w));
    }

    @Override
    public int syevd(char jobz, char uplo, INDArray A, INDArray w) {
        return nativeBlas.syevd(jobz, uplo, pointerConverter.toPointer(A),pointerConverter.toPointer(w));
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z, int[] isuppz) {
        return nativeBlas.syevr(jobz, range, uplo, pointerConverter.toPointer(a), vl, vu, il, iu, abstol, pointerConverter.toPointer(w), pointerConverter.toPointer(z), isuppz);
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, Number abstol, INDArray w, INDArray z, int[] isuppz) {
        return nativeBlas.syevr(jobz, range, uplo, pointerConverter.toPointer(a), vl, vu, il, iu, abstol, pointerConverter.toPointer(w), pointerConverter.toPointer(z), isuppz);
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z, int[] isuppz) {
        return nativeBlas.syevr(jobz, range, uplo, pointerConverter.toPointer(a), vl, vu, il, iu, abstol, pointerConverter.toPointer(w), pointerConverter.toPointer(z), isuppz);
    }

    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z) {
        return nativeBlas.syevx(jobz, range, uplo, pointerConverter.toPointer(a), vl, vu, il, iu, abstol, pointerConverter.toPointer(w), pointerConverter.toPointer(z));
    }

    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a, float vl, float vu, int il, int iu, float abstol, INDArray w, INDArray z) {
        return nativeBlas.syevx(jobz, range, uplo, pointerConverter.toPointer(a), vl, vu, il, iu, abstol, pointerConverter.toPointer(w), pointerConverter.toPointer(z));
    }

    @Override
    public int sygvd(int itype, char jobz, char uplo, INDArray A, INDArray B, INDArray W) {
        return nativeBlas.sygvd(itype, jobz, uplo, pointerConverter.toPointer(A), pointerConverter.toPointer(B), pointerConverter.toPointer(W));
    }

    @Override
    public INDArray sysv(char uplo, INDArray a, int[] ipiv, INDArray b) {
        return nativeBlas.sysv(uplo, pointerConverter.toPointer(a), ipiv, pointerConverter.toPointer(b));
    }


}
