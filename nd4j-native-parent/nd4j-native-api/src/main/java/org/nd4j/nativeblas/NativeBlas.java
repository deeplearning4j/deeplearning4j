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
@Platform(include="NativeBlas.h",link = "libnd4j")
public  class NativeBlas extends Pointer {
    static { Loader.load(); }

    public NativeBlas() {
    }

    public native  IComplexNumber asumComplex(long[] extraPointers,long x);

    public native  double asum(long[] extraPointers,long x);

    public native  INDArray  axpy(long[] extraPointers,double da, long dx, long dy);

    public native  INDArray  axpy(long[] extraPointers,float da, long dx, long dy);

    public native  IComplexNDArray   axpy(long[] extraPointers,IComplexNumber da, long dx, long dy);


    public native  INDArray  axpy(long[] extraPointers,Number da, long dx, long dy);



    public native  INDArray  copy(long[] extraPointers,long x, long y);

    public native  IComplexNDArray  copyComplex(long[] extraPointers,long x, long y);

    public native  double dot(long[] extraPointers,long x, long y);

    public native  IComplexNumber dotc(long[] extraPointers,long x, long y);

    public native  IComplexNumber dotu(long[] extraPointers,long x, long y);

    public native  int geev(long[] extraPointers,char jobvl, char jobvr, long A, long WR, long WI, long VL, long VR);

    public native  void gelsd(long[] extraPointers,long A, long B);

    public native  INDArray  gemm(long[] extraPointers,double alpha, long a, long b, double beta, long c);


    public native  INDArray  gemm(long[] extraPointers,float alpha, long a, long b, float beta, long c);

    public native  IComplexNDArray   gemm(long[] extraPointers,IComplexNumber alpha, long a, long b, IComplexNumber beta, long c);

    public native  INDArray  gemv(long[] extraPointers,double alpha, long a, long x, double beta, long y);

    public native  INDArray  gemv(long[] extraPointers,float alpha, long a, long x, float beta, long y);

    public native  IComplexNDArray   gemv(long[] extraPointers,IComplexDouble alpha, long a, long x, IComplexDouble beta, long y);

    public native  IComplexNDArray   gemv(long[] extraPointers,IComplexFloat alpha, long a, long x, IComplexFloat beta, long y);

    public native  IComplexNDArray   gemv(long[] extraPointers,IComplexNumber alpha, long a, long x, IComplexNumber beta, long y);

    public native  INDArray  gemv(long[] extraPointers,Number alpha, long a, long x, double beta, long y);

    public native  void geqrf(long[] extraPointers,long A, long tau);

    public native  INDArray  ger(long[] extraPointers,double alpha, long x, long y, long a);

    public native  INDArray  ger(long[] extraPointers,float alpha, long x, long y, long a);

    public native  INDArray  ger(long[] extraPointers,Number alpha, long x, long y, long a);

    public native  IComplexNDArray   gerc(long[] extraPointers,IComplexDouble alpha, long x, long y, long a);

    public native  IComplexNDArray   gerc(long[] extraPointers,IComplexFloat alpha, long x, long y, long a);

    public native  IComplexNDArray   geru(long[] extraPointers,IComplexDouble alpha, long x, long y, long a);

    public native  IComplexNDArray   geru(long[] extraPointers,IComplexFloat alpha, long x, long y, long a);

    public native  IComplexNDArray   geru(long[] extraPointers,IComplexNumber alpha, long x, long y, long a);

    public native  INDArray  gesv(long[] extraPointers,long a, int[] ipiv, long b);

    public native  int iamaxComplex(long[] extraPointers,long x);

    public native  int iamax(long[] extraPointers,long x);

    public native  IComplexNumber nrm2Complex(long[] extraPointers,long x);

    public native  double nrm2(long[] extraPointers,long x);

    public native  void ormqr(long[] extraPointers,char side, char trans, long A, long tau, long C);

    public native  void posv(long[] extraPointers,char uplo, long A, long B);

    public native  void saxpy(long[] extraPointers,double alpha, long x, long y);


    public native  void saxpy(long[] extraPointers,float alpha, long x, long y);


    public native  INDArray  scal(long[] extraPointers,double alpha, long x);


    public native  INDArray  scal(long[] extraPointers,float alpha, long x);


    public native  IComplexNDArray  scal(long[] extraPointers,IComplexDouble alpha, long x);

    public native  IComplexNDArray   scal(long[] extraPointers,IComplexFloat alpha, long x);

    public native  IComplexNDArray   scal(long[] extraPointers,IComplexNumber alpha, long x);

    public native  INDArray  swap(long[] extraPointers,long x, long y);

    public native  int syev(long[] extraPointers,char jobz, char uplo, long a, long w);

    public native  int syevd(long[] extraPointers,char jobz, char uplo, long A, long w);

    public native  int syevr(long[] extraPointers,char jobz, char range, char uplo, long a, double vl, double vu, int il, int iu, double abstol, long w, long z, int[] isuppz);


    public native  int syevr(long[] extraPointers,char jobz, char range, char uplo, long a, float vl, float vu, int il, int iu, Number abstol, long w, long z, int[] isuppz);


    public native  int syevr(long[] extraPointers,char jobz, char range, char uplo, long a, float vl, float vu, int il, int iu, float abstol, long w, long z, int[] isuppz);


    public native  int syevx(long[] extraPointers,char jobz, char range, char uplo, long a, double vl, double vu, int il, int iu, double abstol, long w, long z);


    public native  int syevx(long[] extraPointers,char jobz, char range, char uplo, long a, float vl, float vu, int il, int iu, float abstol, long w, long z);


    public native  int sygvd(long[] extraPointers,int itype, char jobz, char uplo, long A, long B, long W);


    public native  INDArray  sysv(long[] extraPointers,char uplo, long a, int[] ipiv, long b);


}
