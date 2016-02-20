package org.nd4j.nativeblas;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Native blas bindings
 * @author Adam Gibson
 */
@Platform(include="NativeBlas.h")
public  class NativeBlas extends Pointer {
    static { Loader.load(); }

    public NativeBlas() {
    }

    public native  IComplexNumber asumComplex(long x);

    public native  double asum(long x);

    public native  INDArray  axpy(double da, long dx, long dy);

    public native  INDArray  axpy(float da, long dx, long dy);

    public native  IComplexNDArray   axpy(IComplexNumber da, long dx, long dy);


    public native  INDArray  axpy(Number da, long dx, long dy);


    public native  void checkInfo(String name, int info);

    public native  INDArray  copy(long x, long y);

    public native  IComplexNDArray  copyComplex(long x, long y);

    public native  double dot(long x, long y);

    public native  IComplexNumber dotc(long x, long y);

    public native  IComplexNumber dotu(long x, long y);

    public native  int geev(char jobvl, char jobvr, long A, long WR, long WI, long VL, long VR);

    public native  void gelsd(long A, long B);

    public native  INDArray  gemm(double alpha, long a, long b, double beta, long c);


    public native  INDArray  gemm(float alpha, long a, long b, float beta, long c);

    public native  IComplexNDArray   gemm(IComplexNumber alpha, long a, long b, IComplexNumber beta, long c);

    public native  INDArray  gemv(double alpha, long a, long x, double beta, long y);

    public native  INDArray  gemv(float alpha, long a, long x, float beta, long y);

    public native  IComplexNDArray   gemv(IComplexDouble alpha, long a, long x, IComplexDouble beta, long y);

    public native  IComplexNDArray   gemv(IComplexFloat alpha, long a, long x, IComplexFloat beta, long y);

    public native  IComplexNDArray   gemv(IComplexNumber alpha, long a, long x, IComplexNumber beta, long y);

    public native  INDArray  gemv(Number alpha, long a, long x, double beta, long y);

    public native  void geqrf(long A, long tau);

    public native  INDArray  ger(double alpha, long x, long y, long a);

    public native  INDArray  ger(float alpha, long x, long y, long a);

    public native  INDArray  ger(Number alpha, long x, long y, long a);

    public native  IComplexNDArray   gerc(IComplexDouble alpha, long x, long y, long a);

    public native  IComplexNDArray   gerc(IComplexFloat alpha, long x, long y, long a);

    public native  IComplexNDArray   geru(IComplexDouble alpha, long x, long y, long a);

    public native  IComplexNDArray   geru(IComplexFloat alpha, long x, long y, long a);

    public native  IComplexNDArray   geru(IComplexNumber alpha, long x, long y, long a);

    public native  INDArray  gesv(long a, int[] ipiv, long b);

    public native  int iamaxComplex(long x);

    public native  int iamax(long x);

    public native  IComplexNumber nrm2Complex(long x);

    public native  double nrm2(long x);

    public native  void ormqr(char side, char trans, long A, long tau, long C);

    public native  void posv(char uplo, long A, long B);

    public native  void saxpy(double alpha, long x, long y);


    public native  void saxpy(float alpha, long x, long y);


    public native  INDArray  scal(double alpha, long x);


    public native  INDArray  scal(float alpha, long x);


    public native  IComplexNDArray   scal(IComplexDouble alpha, long x);

    public native  IComplexNDArray   scal(IComplexFloat alpha, long x);

    public native  IComplexNDArray   scal(IComplexNumber alpha, long x);

    public native  INDArray  swap(long x, long y);

    public native  int syev(char jobz, char uplo, long a, long w);

    public native  int syevd(char jobz, char uplo, long A, long w);

    public native  int syevr(char jobz, char range, char uplo, long a, double vl, double vu, int il, int iu, double abstol, long w, long z, int[] isuppz);


    public native  int syevr(char jobz, char range, char uplo, long a, float vl, float vu, int il, int iu, Number abstol, long w, long z, int[] isuppz);


    public native  int syevr(char jobz, char range, char uplo, long a, float vl, float vu, int il, int iu, float abstol, long w, long z, int[] isuppz);


    public native  int syevx(char jobz, char range, char uplo, long a, double vl, double vu, int il, int iu, double abstol, long w, long z);


    public native  int syevx(char jobz, char range, char uplo, long a, float vl, float vu, int il, int iu, float abstol, long w, long z);


    public native  int sygvd(int itype, char jobz, char uplo, long A, long B, long W);


    public native  INDArray  sysv(char uplo, long a, int[] ipiv, long b);


}
