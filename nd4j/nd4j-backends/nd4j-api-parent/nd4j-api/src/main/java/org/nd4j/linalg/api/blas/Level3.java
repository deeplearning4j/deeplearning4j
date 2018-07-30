/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.blas;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 
  Level 3 blas implementations.
  Incx and other parameters are inferred
  from the given ndarrays.
 
  To avoid boxing, doubles are used in place of normal numbers.
  The underlying implementation will call the proper data opType.
 
  This is a fortran 95 style api that gives us the efficiency
  and flexibility of the fortran 77 api
 
  Credit to:
  https://www.ualberta.ca/AICT/RESEARCH/LinuxClusters/doc/mkl81/mklqref/blaslev3.htm
 
  for the descriptions
 
  @author Adam Gibson
*/
public interface Level3 {
    /**
     * gemm performs a matrix-matrix operation
     c := alpha*op(a)*op(b) + beta*c,
     where c is an m-by-n matrix,
     op(a) is an m-by-k matrix,
     op(b) is a k-by-n matrix.
     * @param Order
     * @param TransA
     * @param TransB
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    void gemm(char Order, char TransA, char TransB, double alpha, INDArray A, INDArray B, double beta, INDArray C);

    /** A convenience method for matrix-matrix operations with transposes.
     * Implements C = alpha*op(A)*op(B) + beta*C
     * Matrices A and B can be any order and offset (though will have copy overhead if elements are not contiguous in buffer)
     * but matrix C MUST be f order, 0 offset and have length == data.length
     */
    void gemm(INDArray A, INDArray B, INDArray C, boolean transposeA, boolean transposeB, double alpha, double beta);


    /**
     * her2k performs a rank-2k update of an n-by-n Hermitian matrix c, that is, one of the following operations:
     c := alpha*a*conjg(b') + conjg(alpha)*b*conjg(a') + beta*c,  for trans = 'N'or'n'
     c := alpha*conjg(b')*a + conjg(alpha)*conjg(a')*b + beta*c,  for trans = 'C'or'c'
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
    void symm(char Order, char Side, char Uplo, double alpha, INDArray A, INDArray B, double beta, INDArray C);

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
    void syrk(char Order, char Uplo, char Trans, double alpha, INDArray A, double beta, INDArray C);

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
    void syr2k(char Order, char Uplo, char Trans, double alpha, INDArray A, INDArray B, double beta, INDArray C);

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
    void trmm(char Order, char Side, char Uplo, char TransA, char Diag, double alpha, INDArray A, INDArray B,
                    INDArray C);

    /**
     * ?trsm solves one of the following matrix equations:
     op(a)*x = alpha*b  or  x*op(a) = alpha*b,
     where x and b are m-by-n general matrices, and a is triangular;
     op(a) must be an m-by-m matrix, if side = 'L'or'l'
     op(a) must be an n-by-n matrix, if side = 'R'or'r'.
     For the definition of op(a), see Matrix Arguments.
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
    void trsm(char Order, char Side, char Uplo, char TransA, char Diag, double alpha, INDArray A, INDArray B);


    /**
     * gemm performs a matrix-matrix operation
     c := alpha*op(a)*op(b) + beta*c,
     where c is an m-by-n matrix,
     op(a) is an m-by-k matrix,
     op(b) is a k-by-n matrix.
     * @param Order
     * @param TransA
     * @param TransB
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    void gemm(char Order, char TransA, char TransB, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C);

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
    void hemm(char Order, char Side, char Uplo, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C);

    /**
     * herk performs a rank-n update of a Hermitian matrix, that is, one of the following operations:
     c := alpha*a*conjug(a') + beta*c  for trans = 'N'or'n'
     c := alpha*conjug(a')*a + beta*c  for trans = 'C'or'c',
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
    void herk(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNumber beta,
                    IComplexNDArray C);

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
    void her2k(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C);

    /**
     * her2k performs a rank-2k update of an n-by-n Hermitian matrix c, that is, one of the following operations:
     c := alpha*a*conjg(b') + conjg(alpha)*b*conjg(a') + beta*c,  for trans = 'N'or'n'
     c := alpha*conjg(b')*a + conjg(alpha)*conjg(a')*b + beta*c,  for trans = 'C'or'c'
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
    void symm(char Order, char Side, char Uplo, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C);

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
    void syrk(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNumber beta,
                    IComplexNDArray C);

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
    void syr2k(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C);

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
    void trmm(char Order, char Side, char Uplo, char TransA, char Diag, IComplexNumber alpha, IComplexNDArray A,
                    IComplexNDArray B, IComplexNDArray C);

    /**
     * ?trsm solves one of the following matrix equations:
     op(a)*x = alpha*b  or  x*op(a) = alpha*b,
     where x and b are m-by-n general matrices, and a is triangular;
     op(a) must be an m-by-m matrix, if side = 'L'or'l'
     op(a) must be an n-by-n matrix, if side = 'R'or'r'.
     For the definition of op(a), see Matrix Arguments.
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
    void trsm(char Order, char Side, char Uplo, char TransA, char Diag, IComplexNumber alpha, IComplexNDArray A,
                    IComplexNDArray B);


}
