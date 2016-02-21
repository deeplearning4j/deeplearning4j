package org.nd4j.nativeblas;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Native bindings Level 2
 * @author Adam Gibson
 */
@Platform(include="NativeLevel2.h",link = "libnd4j")
public class NativeLevel2 extends Pointer {
    /**
     * gemv computes a matrix-vector product using a general matrix and performs one of the following matrix-vector operations: 
     y := alpha*a*x + beta*y  for trans = 'N'or'n'; 
     y := alpha*a'*x + beta*y  for trans = 'T'or't'; 
     y := alpha*conjg(long[] extraPointers,a')*x + beta*y  for trans = 'C'or'c'. 
     Here a is an m-by-n band matrix, x and y are vectors, alpha and beta are scalars.
     * @param order
     * @param transA
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    public native void gemv(long[] extraPointers, char order,
               char transA,
               double alpha,  long A,
               long X, double beta,
               long Y);

    /**
     * gemv computes a matrix-vector product using a general matrix and performs one of the following matrix-vector operations:
     y := alpha*a*x + beta*y  for trans = 'N'or'n';
     y := alpha*a'*x + beta*y  for trans = 'T'or't';
     y := alpha*conjg(long[] extraPointers,a')*x + beta*y  for trans = 'C'or'c'.
     Here a is an m-by-n band matrix, x and y are vectors, alpha and beta are scalars.
     * @param order
     * @param transA
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    public native void gemv(long[] extraPointers, char order,
               char transA,
               IComplexNumber alpha,  long A,
               long X, IComplexNumber beta,
               long Y);

    /**
     * gbmv computes a matrix-vector product using a general band matrix and performs one of the following matrix-vector operations:
     y := alpha*a*x + beta*y  for trans = 'N'or'n';
     y := alpha*a'*x + beta*y  for trans = 'T'or't';
     y := alpha*conjg(long[] extraPointers,a')*x + beta*y  for trans = 'C'or'c'.
     Here a is an m-by-n band matrix with ku superdiagonals and kl subdiagonals, x and y are vectors, alpha and beta are scalars.
     * @param order
     * @param TransA
     * @param KL
     * @param KU
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    public native void gbmv(long[] extraPointers,char order,
              char TransA,
              int KL,  int KU,  double alpha,
              long A,  long X,
              double beta, long Y);

    /**
     * gbmv computes a matrix-vector product using a general band matrix and performs one of the following matrix-vector operations:
     y := alpha*a*x + beta*y  for trans = 'N'or'n';
     y := alpha*a'*x + beta*y  for trans = 'T'or't';
     y := alpha*conjg(long[] extraPointers,a')*x + beta*y  for trans = 'C'or'c'.
     Here a is an m-by-n band matrix with ku superdiagonals and kl subdiagonals, x and y are vectors, alpha and beta are scalars.
     * @param order
     * @param TransA
     * @param KL
     * @param KU
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    public native void gbmv(long[] extraPointers,char order,
              char TransA,
              int KL,  int KU,  IComplexNumber alpha,
              long A,  long X,
              IComplexNumber beta, long Y);


    /**
     *  performs a rank-1 update of a general m-by-n matrix a:
     a := alpha*x*y' + a.
     * @param order
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    public native void ger(long[] extraPointers, char order,
              double alpha,  long X,
              long Y, long A);



    /**
     * performs a rank-1 update of a general m-by-n matrix a, without conjugation:
     a := alpha*x*y' + a.
     * @param order
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    public native void geru(long[] extraPointers, char order,
               IComplexNumber alpha,  long X,
               long Y, long A);




    /**
     * performs a rank-1 update of a general m-by-n matrix a, without conjugation:
     a := alpha*x*y' + a.
     * @param order
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    public native void hbmv(long[] extraPointers,char order, char Uplo, IComplexNumber alpha, long A, long X, IComplexNumber beta, long Y);
    /**
     * hemv computes a matrix-vector product using a Hermitian matrix:
     y := alpha*a*x + beta*y.
     Here a is an n-by-n Hermitian band matrix with k superdiagonals, x and y are n-element vectors, alpha and beta are scalars.
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    public native void hemv(long[] extraPointers,char order, char Uplo, IComplexNumber alpha, long A, long X, IComplexNumber beta, long Y);
    /**
     * ?her2 performs a rank-2 update of an n-by-n Hermitian matrix a:
     a := alpha*x*conjg(long[] extraPointers,y') + conjg(long[] extraPointers,alpha)*y*conjg(long[] extraPointers,x') + a.
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    public native void her2(long[] extraPointers,char order, char Uplo, IComplexNumber alpha, long X, long Y, long A);
    /**
     * ?hpmv computes a matrix-vector product using a Hermitian packed matrix:
     y := alpha*a*x + beta*y.
     Here a is an n-by-n packed Hermitian matrix, x and y are n-element vectors, alpha and beta are scalars.
     * @param order
     * @param Uplo
     * @param N
     * @param alpha
     * @param Ap
     * @param X
     * @param beta
     * @param Y
     */
    public native void hpmv(long[] extraPointers,char order, char Uplo, int N, IComplexNumber alpha, long Ap, long X, IComplexNumber beta, long Y);

    /**
     * hpr2 performs a rank-2 update of an n-by-n packed Hermitian matrix a:
     a := alpha*x*conjg(long[] extraPointers,y') + conjg(long[] extraPointers,alpha)*y*conjg(long[] extraPointers,x') + a.
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param Ap
     */
    public native void hpr2(long[] extraPointers,char order, char Uplo, IComplexNumber alpha, long X, long Y, long Ap);

    /**
     * sbmv computes a matrix-vector product using a symmetric band matrix:
     y := alpha*a*x + beta*y.
     Here a is an n-by-n symmetric band matrix with k superdiagonals, x and y are n-element vectors, alpha and beta are scalars.
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    public native void sbmv(long[] extraPointers, char order,  char Uplo,
               double alpha,  long A,
               long X,
               double beta, long Y);

    /**
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param Ap
     * @param X
     * @param beta
     * @param Y
     */
    public native void spmv(long[] extraPointers, char order,  char Uplo,
               double alpha,  long Ap,
               long X,
               double beta, long Y);

    /**
     * spr performs a rank-1 update of an n-by-n packed symmetric matrix a:
     a := alpha*x*x' + a. 
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Ap
     */
    public native void spr(long[] extraPointers, char order,  char Uplo,
              double alpha,  long X,
              long Ap);

    /**
     * ?spr2 performs a rank-2 update of an n-by-n packed symmetric matrix a:
     a := alpha*x*y' + alpha*y*x' + a. 
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    public native void spr2(long[] extraPointers,char order, char Uplo,
              double alpha, long X,
              long Y,long A);

    /**
     * symv computes a matrix-vector product for a symmetric matrix:
     y := alpha*a*x + beta*y.
     Here a is an n-by-n symmetric matrix; x and y are n-element vectors, alpha and beta are scalars.
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    public native void symv(long[] extraPointers,char order, char Uplo,
              double alpha, long A,
              long X,
              double beta, long Y);

    /**
     * syr performs a rank-1 update of an n-by-n symmetric matrix a:
     a := alpha*x*x' + a.
     * @param order
     * @param Uplo
     * @param N
     * @param alpha
     * @param X
     * @param A
     */
    public native void syr(long[] extraPointers, char order,  char Uplo,
              int N,  double alpha,  long X,
              long A);

    /**
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    public native void syr2(long[] extraPointers, char order,  char Uplo,
               double alpha,  long X,
               long Y, long A);

    /**
     * syr2 performs a rank-2 update of an n-by-n symmetric matrix a:
     a := alpha*x*y' + alpha*y*x' + a.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    public native void tbmv(long[] extraPointers, char order,  char Uplo,
               char  TransA,  char  Diag,
               long A,
               long X);

    /**
     * ?tbsv solves a system of linear equations whose coefficients are in a triangular band matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    public native void tbsv(long[] extraPointers, char order,  char Uplo,
               char  TransA,  char  Diag,
               long A,
               long X);

    /**
     * tpmv computes a matrix-vector product using a triangular packed matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param Ap
     * @param X
     */
    public native void tpmv(long[] extraPointers, char order,  char Uplo,
               char  TransA,  char  Diag,
               long Ap, long X);

    /**
     * tpsv solves a system of linear equations whose coefficients are in a triangular packed matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param Ap
     * @param X
     */
    public native void tpsv(long[] extraPointers, char order,  char Uplo,
               char  TransA,  char  Diag,
               long Ap, long X);

    /**
     * trmv computes a matrix-vector product using a triangular matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    public native void trmv(long[] extraPointers, char order,  char Uplo,
               char  TransA,  char  Diag,
               long A,
               long X);

    /**
     * trsv solves a system of linear equations whose coefficients are in a triangular matrix.
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    public native void trsv(long[] extraPointers, char order,  char Uplo,
               char  TransA,  char  Diag,
               long A,long X);
}
