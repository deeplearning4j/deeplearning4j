package org.nd4j.nativeblas;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Native bindings for level 3
 * @author Adam Gibson
 */
@Platform(include="NativeLevel3.h",link = "libnd4j")
public class NativeLevel3 extends Pointer {
    /**
     * gemm performs a matrix-matrix operation
     c := alpha*op(long[] extraPointers,a)*op(long[] extraPointers,b) + beta*c,
     where c is an m-by-n matrix,
     op(long[] extraPointers,a) is an m-by-k matrix,
     op(long[] extraPointers,b) is a k-by-n matrix.
     * @param Order
     * @param TransA
     * @param TransB
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    public native void gemm(long[] extraPointers, char Order,  char TransA,
               char TransB,
               double alpha,  long  A,
               long  B,
               double beta, long  C);

    /** A convenience method for matrix-matrix operations with transposes.
     * Implements C = alpha*op(long[] extraPointers,A)*op(long[] extraPointers,B) + beta*C
     * Matrices A and B can be any order and offset (long[] extraPointers,though will have copy overhead if elements are not contiguous in buffer)
     * but matrix C MUST be f order, 0 offset and have length == data.length
     */
    public native void gemm(long[] extraPointers,long  A, long  B, long  C, boolean transposeA, boolean transposeB, double alpha, double beta );


    /**
     * her2k performs a rank-2k update of an n-by-n Hermitian matrix c, that is, one of the following operations:
     c := alpha*a*conjg(long[] extraPointers,b') + conjg(long[] extraPointers,alpha)*b*conjg(long[] extraPointers,a') + beta*c,  for trans = 'N'or'n'
     c := alpha*conjg(long[] extraPointers,b')*a + conjg(long[] extraPointers,alpha)*conjg(long[] extraPointers,a')*b + beta*c,  for trans = 'C'or'c'
     where c is an n-by-n Hermitian matrix;
     a and b are n-by-k matrices if trans = 'N'or'n',
     a and b are k-by-n matrices if trans = 'C'or'c'.
     * @param Order
     * @param Side
     * @param Uplo
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    public native void symm(long[] extraPointers, char Order,  char Side,
               char Uplo,
               double alpha,  long  A,
               long  B, double beta,
               long  C);

    /**
     * syrk performs a rank-n update of an n-by-n symmetric matrix c, that is, one of the following operations:
     c := alpha*a*a' + beta*c  for trans = 'N'or'n'
     c := alpha*a'*a + beta*c  for trans = 'T'or't','C'or'c',
     where c is an n-by-n symmetric matrix;
     a is an n-by-k matrix, if trans = 'N'or'n',
     a is a k-by-n matrix, if trans = 'T'or't','C'or'c'.
     * @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param beta
     * @param C
     */
    public native void syrk(long[] extraPointers, char Order,  char Uplo,
               char Trans,
               double alpha,  long  A,
               double beta, long  C);

    /**
     * yr2k performs a rank-2k update of an n-by-n symmetric matrix c, that is, one of the following operations:
     c := alpha*a*b' + alpha*b*a' + beta*c  for trans = 'N'or'n'
     c := alpha*a'*b + alpha*b'*a + beta*c  for trans = 'T'or't',
     where c is an n-by-n symmetric matrix;
     a and b are n-by-k matrices, if trans = 'N'or'n',
     a and b are k-by-n matrices, if trans = 'T'or't'.
     * @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    public native void syr2k(long[] extraPointers, char Order,  char Uplo,
                char Trans,
                double alpha,  long  A,
                long  B, double beta,
                long  C);

    /**
     * syr2k performs a rank-2k update of an n-by-n symmetric matrix c, that is, one of the following operations:
     c := alpha*a*b' + alpha*b*a' + beta*c  for trans = 'N'or'n'
     c := alpha*a'*b + alpha*b'*a + beta*c  for trans = 'T'or't',
     where c is an n-by-n symmetric matrix;
     a and b are n-by-k matrices, if trans = 'N'or'n',
     a and b are k-by-n matrices, if trans = 'T'or't'.
     * @param Order
     * @param Side
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param alpha
     * @param A
     * @param B
     * @param C
     */
    public native void trmm(long[] extraPointers,char Order, char Side,
              char Uplo, char TransA,
              char Diag,
              double alpha, long  A,
              long  B, long  C);

    /**
     * ?trsm solves one of the following matrix equations:
     op(long[] extraPointers,a)*x = alpha*b  or  x*op(long[] extraPointers,a) = alpha*b,
     where x and b are m-by-n general matrices, and a is triangular;
     op(long[] extraPointers,a) must be an m-by-m matrix, if side = 'L'or'l'
     op(long[] extraPointers,a) must be an n-by-n matrix, if side = 'R'or'r'.
     For the definition of op(long[] extraPointers,a), see Matrix Arguments.
     The routine overwrites x on b.
     * @param Order
     * @param Side
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param alpha
     * @param A
     * @param B
     */
    public native void trsm(long[] extraPointers, char Order,  char Side,
               char Uplo,  char TransA,
               char Diag,
               double alpha,  long  A,
               long  B);


    /**
     * gemm performs a matrix-matrix operation
     c := alpha*op(long[] extraPointers,a)*op(long[] extraPointers,b) + beta*c,
     where c is an m-by-n matrix,
     op(long[] extraPointers,a) is an m-by-k matrix,
     op(long[] extraPointers,b) is a k-by-n matrix.
     * @param Order
     * @param TransA
     * @param TransB
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    public native void gemm(long[] extraPointers, char Order,  char TransA,
               char TransB,
               IComplexNumber alpha,  long  A,
               long  B,
               IComplexNumber beta, long  C);

    /**
     * hemm performs one of the following matrix-matrix operations:
     c := alpha*a*b + beta*c  for side = 'L'or'l'
     c := alpha*b*a + beta*c  for side = 'R'or'r',
     where a is a Hermitian matrix,
     b and c are m-by-n matrices.
     * @param Order
     * @param Side
     * @param Uplo
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    public native void hemm(long[] extraPointers, char Order,  char Side,
               char Uplo,
               IComplexNumber alpha,  long  A,
               long  B,IComplexNumber beta,
               long  C);

    /**
     * herk performs a rank-n update of a Hermitian matrix, that is, one of the following operations:
     c := alpha*a*conjug(long[] extraPointers,a') + beta*c  for trans = 'N'or'n'
     c := alpha*conjug(long[] extraPointers,a')*a + beta*c  for trans = 'C'or'c',
     where c is an n-by-n Hermitian matrix;
     a is an n-by-k matrix, if trans = 'N'or'n',
     a is a k-by-n matrix, if trans = 'C'or'c'.
     * @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param beta
     * @param C
     */
    public native void herk(long[] extraPointers, char Order,  char Uplo,
               char Trans,
               IComplexNumber alpha,  long  A,
               IComplexNumber beta, long  C);

    /**
     *  @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    public native void her2k(long[] extraPointers, char Order,  char Uplo,
                char Trans,
                IComplexNumber alpha,  long  A,
                long  B, IComplexNumber beta,
                long  C);

    /**
     * her2k performs a rank-2k update of an n-by-n Hermitian matrix c, that is, one of the following operations:
     c := alpha*a*conjg(long[] extraPointers,b') + conjg(long[] extraPointers,alpha)*b*conjg(long[] extraPointers,a') + beta*c,  for trans = 'N'or'n'
     c := alpha*conjg(long[] extraPointers,b')*a + conjg(long[] extraPointers,alpha)*conjg(long[] extraPointers,a')*b + beta*c,  for trans = 'C'or'c'
     where c is an n-by-n Hermitian matrix;
     a and b are n-by-k matrices if trans = 'N'or'n',
     a and b are k-by-n matrices if trans = 'C'or'c'.
     * @param Order
     * @param Side
     * @param Uplo
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    public native void symm(long[] extraPointers, char Order,  char Side,
               char Uplo,
               IComplexNumber alpha,  long  A,
               long  B, IComplexNumber beta,
               long  C);

    /**
     * syrk performs a rank-n update of an n-by-n symmetric matrix c, that is, one of the following operations:
     c := alpha*a*a' + beta*c  for trans = 'N'or'n'
     c := alpha*a'*a + beta*c  for trans = 'T'or't','C'or'c',
     where c is an n-by-n symmetric matrix;
     a is an n-by-k matrix, if trans = 'N'or'n',
     a is a k-by-n matrix, if trans = 'T'or't','C'or'c'.
     * @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param beta
     * @param C
     */
    public native void syrk(long[] extraPointers, char Order,  char Uplo,
               char Trans,
               IComplexNumber alpha,  long  A,
               IComplexNumber beta, long  C);

    /**
     * yr2k performs a rank-2k update of an n-by-n symmetric matrix c, that is, one of the following operations:
     c := alpha*a*b' + alpha*b*a' + beta*c  for trans = 'N'or'n'
     c := alpha*a'*b + alpha*b'*a + beta*c  for trans = 'T'or't',
     where c is an n-by-n symmetric matrix;
     a and b are n-by-k matrices, if trans = 'N'or'n',
     a and b are k-by-n matrices, if trans = 'T'or't'.
     * @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    public native void syr2k(long[] extraPointers, char Order,  char Uplo,
                char Trans,
                IComplexNumber alpha,  long  A,
                long  B,IComplexNumber beta,
                long  C);

    /**
     * syr2k performs a rank-2k update of an n-by-n symmetric matrix c, that is, one of the following operations:
     c := alpha*a*b' + alpha*b*a' + beta*c  for trans = 'N'or'n'
     c := alpha*a'*b + alpha*b'*a + beta*c  for trans = 'T'or't',
     where c is an n-by-n symmetric matrix;
     a and b are n-by-k matrices, if trans = 'N'or'n',
     a and b are k-by-n matrices, if trans = 'T'or't'.
     * @param Order
     * @param Side
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param alpha
     * @param A
     * @param B
     * @param C
     */
    public native void trmm(long[] extraPointers,char Order, char Side,
              char Uplo, char TransA,
              char Diag,
              IComplexNumber alpha, long  A,
              long  B, long  C);

    /**
     * ?trsm solves one of the following matrix equations:
     op(long[] extraPointers,a)*x = alpha*b  or  x*op(long[] extraPointers,a) = alpha*b,
     where x and b are m-by-n general matrices, and a is triangular;
     op(long[] extraPointers,a) must be an m-by-m matrix, if side = 'L'or'l'
     op(long[] extraPointers,a) must be an n-by-n matrix, if side = 'R'or'r'.
     For the definition of op(long[] extraPointers,a), see Matrix Arguments.
     The routine overwrites x on b.
     * @param Order
     * @param Side
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param alpha
     * @param A
     * @param B
     */
    public native void trsm(long[] extraPointers, char Order,  char Side,
               char Uplo,  char TransA,
               char Diag,
               IComplexNumber alpha,  long  A,
               long  B);



}
