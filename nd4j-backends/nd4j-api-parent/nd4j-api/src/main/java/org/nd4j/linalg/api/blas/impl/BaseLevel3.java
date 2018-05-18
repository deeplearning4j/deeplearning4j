package org.nd4j.linalg.api.blas.impl;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.blas.params.GemmParams;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;

/**
 * Base class for level 3 functions, abstract headers pulled from:
 * http://www.netlib.org/blas/blast-forum/cblas.h
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseLevel3 extends BaseLevel implements Level3 {
    /**
     * gemm performs a matrix-matrix operation
     * c := alpha*op(a)*op(b) + beta*c,
     * where c is an m-by-n matrix,
     * op(a) is an m-by-k matrix,
     * op(b) is a k-by-n matrix.
     *  @param Order
     * @param TransA
     * @param TransB
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    @Override
    public void gemm(char Order, char TransA, char TransB, double alpha, INDArray A, INDArray B, double beta,
                    INDArray C) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(true, A, B, C);

        GemmParams params = new GemmParams(A, B, C);

        int charOder = Order;
        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, params.getA(), params.getB(), params.getC());
            dgemm(Order, params.getTransA(), params.getTransB(), params.getM(), params.getN(), params.getK(), 1.0,
                            params.getA(), params.getLda(), params.getB(), params.getLdb(), 0, C, params.getLdc());
        } else if (A.data().dataType() == DataBuffer.Type.FLOAT) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, params.getA(), params.getB(), params.getC());
            sgemm(Order, params.getTransA(), params.getTransB(), params.getM(), params.getN(), params.getK(), 1.0f,
                            params.getA(), params.getLda(), params.getB(), params.getLdb(), 0, C, params.getLdc());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, params.getA(), params.getB(), params.getC());
            hgemm(Order, params.getTransA(), params.getTransB(), params.getM(), params.getN(), params.getK(), 1.0f,
                            params.getA(), params.getLda(), params.getB(), params.getLdb(), 0, C, params.getLdc());
        }

        OpExecutionerUtil.checkForAny(C);
    }

    /**{@inheritDoc}
     */
    @Override
    public void gemm(INDArray A, INDArray B, INDArray C, boolean transposeA, boolean transposeB, double alpha,
                    double beta) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(true, A, B, C);

        GemmParams params = new GemmParams(A, B, C, transposeA, transposeB);
        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, params.getA(), params.getB(), C);
            dgemm(A.ordering(), params.getTransA(), params.getTransB(), params.getM(), params.getN(), params.getK(),
                            alpha, params.getA(), params.getLda(), params.getB(), params.getLdb(), beta, C,
                            params.getLdc());
        } else if (A.data().dataType() == DataBuffer.Type.FLOAT) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, params.getA(), params.getB(), C);
            sgemm(A.ordering(), params.getTransA(), params.getTransB(), params.getM(), params.getN(), params.getK(),
                            (float) alpha, params.getA(), params.getLda(), params.getB(), params.getLdb(), (float) beta,
                            C, params.getLdc());
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, params.getA(), params.getB(), C);
            hgemm(A.ordering(), params.getTransA(), params.getTransB(), params.getM(), params.getN(), params.getK(),
                            (float) alpha, params.getA(), params.getLda(), params.getB(), params.getLdb(), (float) beta,
                            C, params.getLdc());
        }

        OpExecutionerUtil.checkForAny(C);
    }


    /**
     * her2k performs a rank-2k update of an n-by-n Hermitian matrix c, that is, one of the following operations:
     * c := alpha*a*conjg(b') + conjg(alpha)*b*conjg(a') + beta*c,  for trans = 'N'or'n'
     * c := alpha*conjg(b')*a + conjg(alpha)*conjg(a')*b + beta*c,  for trans = 'C'or'c'
     * where c is an n-by-n Hermitian matrix;
     * a and b are n-by-k matrices if trans = 'N'or'n',
     * a and b are k-by-n matrices if trans = 'C'or'c'.
     *  @param Order
     * @param Side
     * @param Uplo
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    @Override
    public void symm(char Order, char Side, char Uplo, double alpha, INDArray A, INDArray B, double beta, INDArray C) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, B, C);

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, B, C);
            dsymm(Order, Side, Uplo, (int) C.rows(), (int) C.columns(), alpha, A, (int) A.size(0), B, (int) B.size(0), beta, C, (int) C.size(0));
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, B, C);
            ssymm(Order, Side, Uplo, (int) C.rows(), (int) C.columns(), (float) alpha, A, (int) A.size(0), B, (int) B.size(0), (float) beta, C,
                    (int) C.size(0));
        }

        OpExecutionerUtil.checkForAny(C);
    }

    /**
     * syrk performs a rank-n update of an n-by-n symmetric matrix c, that is, one of the following operations:
     * c := alpha*a*a' + beta*c  for trans = 'N'or'n'
     * c := alpha*a'*a + beta*c  for trans = 'T'or't','C'or'c',
     * where c is an n-by-n symmetric matrix;
     * a is an n-by-k matrix, if trans = 'N'or'n',
     * a is a k-by-n matrix, if trans = 'T'or't','C'or'c'.
     *  @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param beta
     * @param C
     */
    @Override
    public void syrk(char Order, char Uplo, char Trans, double alpha, INDArray A, double beta, INDArray C) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, C);

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, C);
            dsyrk(Order, Uplo, Trans, (int) C.rows(), 1, alpha, A, (int) A.size(0), beta, C, (int) C.size(0));
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, C);
            ssyrk(Order, Uplo, Trans, (int) C.rows(), 1, (float) alpha, A, (int) A.size(0), (float) beta, C, (int) C.size(0));
        }

        OpExecutionerUtil.checkForAny(C);
    }

    /**
     * yr2k performs a rank-2k update of an n-by-n symmetric matrix c, that is, one of the following operations:
     * c := alpha*a*b' + alpha*b*a' + beta*c  for trans = 'N'or'n'
     * c := alpha*a'*b + alpha*b'*a + beta*c  for trans = 'T'or't',
     * where c is an n-by-n symmetric matrix;
     * a and b are n-by-k matrices, if trans = 'N'or'n',
     * a and b are k-by-n matrices, if trans = 'T'or't'.
     *  @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    @Override
    public void syr2k(char Order, char Uplo, char Trans, double alpha, INDArray A, INDArray B, double beta,
                    INDArray C) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, B, C);

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, B, C);
            dsyr2k(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), alpha, A, (int) A.size(0), B, (int) B.size(0), beta, C, (int) C.size(0));
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, B, C);
            ssyr2k(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), (float) alpha, A, (int) A.size(0), B, (int) B.size(0), (float) beta, C, (int) C.size(0));
        }

        OpExecutionerUtil.checkForAny(C);
    }

    /**
     * syr2k performs a rank-2k update of an n-by-n symmetric matrix c, that is, one of the following operations:
     * c := alpha*a*b' + alpha*b*a' + beta*c  for trans = 'N'or'n'
     * c := alpha*a'*b + alpha*b'*a + beta*c  for trans = 'T'or't',
     * where c is an n-by-n symmetric matrix;
     * a and b are n-by-k matrices, if trans = 'N'or'n',
     * a and b are k-by-n matrices, if trans = 'T'or't'.
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
    @Override
    public void trmm(char Order, char Side, char Uplo, char TransA, char Diag, double alpha, INDArray A, INDArray B,
                    INDArray C) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, B, C);

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, B, C);
            dtrmm(Order, Side, Uplo, TransA, Diag, (int) A.rows(), (int) A.columns(), alpha, A, (int) A.size(0), B, (int) B.size(0));
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, B, C);
            strmm(Order, Side, Uplo, TransA, Diag, (int) A.rows(), (int) A.columns(), (float) alpha, A, (int) A.size(0), B, (int) B.size(0));
        }

        OpExecutionerUtil.checkForAny(C);
    }

    /**
     * ?trsm solves one of the following matrix equations:
     * op(a)*x = alpha*b  or  x*op(a) = alpha*b,
     * where x and b are m-by-n general matrices, and a is triangular;
     * op(a) must be an m-by-m matrix, if side = 'L'or'l'
     * op(a) must be an n-by-n matrix, if side = 'R'or'r'.
     * For the definition of op(a), see Matrix Arguments.
     * The routine overwrites x on b.
     *  @param Order
     * @param Side
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param alpha
     * @param A
     * @param B
     */
    @Override
    public void trsm(char Order, char Side, char Uplo, char TransA, char Diag, double alpha, INDArray A, INDArray B) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(false, A, B);

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, A, B);
            dtrsm(Order, Side, Uplo, TransA, Diag, (int) A.rows(), (int) A.columns(), alpha, A, (int) A.size(0), B, (int) B.size(0));
        } else {
            DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, A, B);
            strsm(Order, Side, Uplo, TransA, Diag, (int) A.rows(), (int) A.columns(), (float) alpha, A, (int) A.size(0), B, (int) B.size(0));
        }

        OpExecutionerUtil.checkForAny(B);
    }

    /**
     * gemm performs a matrix-matrix operation
     * c := alpha*op(a)*op(b) + beta*c,
     * where c is an m-by-n matrix,
     * op(a) is an m-by-k matrix,
     * op(b) is a k-by-n matrix.
     *  @param Order
     * @param TransA
     * @param TransB
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    @Override
    public void gemm(char Order, char TransA, char TransB, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {
        if (Nd4j.getExecutioner().getProfilingMode() == OpExecutioner.ProfilingMode.ALL)
            OpProfiler.getInstance().processBlasCall(true, A, B, C);

        GemmParams params = new GemmParams(A, B, C);

        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            zgemm(Order, TransA, TransB, params.getM(), params.getN(), params.getK(), alpha.asDouble(),
                            A.ordering() == NDArrayFactory.C ? B : A, params.getLda(),
                            B.ordering() == NDArrayFactory.C ? A : B, params.getLdb(), beta.asDouble(), C,
                            params.getLdc());
        } else
            cgemm(Order, TransA, TransB, params.getM(), params.getN(), params.getK(), alpha.asFloat(),
                            A.ordering() == NDArrayFactory.C ? B : A, params.getLda(),
                            B.ordering() == NDArrayFactory.C ? A : B, params.getLdb(), beta.asFloat(), C,
                            params.getLdc());

    }

    /**
     * hemm performs one of the following matrix-matrix operations:
     * c := alpha*a*b + beta*c  for side = 'L'or'l'
     * c := alpha*b*a + beta*c  for side = 'R'or'r',
     * where a is a Hermitian matrix,
     * b and c are m-by-n matrices.
     *  @param Order
     * @param Side
     * @param Uplo
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    @Override
    public void hemm(char Order, char Side, char Uplo, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            zhemm(Order, Side, Uplo, (int) B.rows(), (int) B.columns(), alpha.asDouble(), A, (int) A.size(0), B, (int) B.size(0), beta.asDouble(), C, (int) C.size(0));
        else
            chemm(Order, Side, Uplo, (int) B.rows(), (int) B.columns(), alpha.asFloat(), A, (int) A.size(0), B, (int) B.size(0), beta.asFloat(), C, (int) C.size(0));
    }

    /**
     * herk performs a rank-n update of a Hermitian matrix, that is, one of the following operations:
     * c := alpha*a*conjug(a') + beta*c  for trans = 'N'or'n'
     * c := alpha*conjug(a')*a + beta*c  for trans = 'C'or'c',
     * where c is an n-by-n Hermitian matrix;
     * a is an n-by-k matrix, if trans = 'N'or'n',
     * a is a k-by-n matrix, if trans = 'C'or'c'.
     *  @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param beta
     * @param C
     */
    @Override
    public void herk(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNumber beta,
                    IComplexNDArray C) {
        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            zherk(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), alpha.asDouble(), A, (int) A.size(0), beta.asDouble(), C, (int) C.size(0));
        else
            cherk(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), alpha.asFloat(), A, (int) A.size(0), beta.asFloat(), C, (int) C.size(0));

    }

    /**
     * @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    @Override
    public void her2k(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {
        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            zher2k(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), alpha.asDouble(), A, (int) A.size(0), B, (int) B.size(0), beta.asDouble(), C, (int) C.size(0));
        else
            cher2k(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), alpha.asFloat(), A, (int) A.size(0), B, (int) B.size(0), beta.asFloat(), C, (int) C.size(0));

    }

    /**
     * her2k performs a rank-2k update of an n-by-n Hermitian matrix c, that is, one of the following operations:
     * c := alpha*a*conjg(b') + conjg(alpha)*b*conjg(a') + beta*c,  for trans = 'N'or'n'
     * c := alpha*conjg(b')*a + conjg(alpha)*conjg(a')*b + beta*c,  for trans = 'C'or'c'
     * where c is an n-by-n Hermitian matrix;
     * a and b are n-by-k matrices if trans = 'N'or'n',
     * a and b are k-by-n matrices if trans = 'C'or'c'.
     *  @param Order
     * @param Side
     * @param Uplo
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    @Override
    public void symm(char Order, char Side, char Uplo, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {
        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            zsymm(Order, Side, Uplo, (int) A.rows(), (int) A.columns(), alpha.asDouble(), A, (int) A.size(0), B, (int) B.size(0), beta.asDouble(), C, (int) C.size(0));
        else
            csymm(Order, Side, Uplo, (int) A.rows(), (int) A.columns(), alpha.asFloat(), A, (int) A.size(0), B, (int) B.size(0), beta.asFloat(), C, (int) C.size(0));

    }

    /**
     * syrk performs a rank-n update of an n-by-n symmetric matrix c, that is, one of the following operations:
     * c := alpha*a*a' + beta*c  for trans = 'N'or'n'
     * c := alpha*a'*a + beta*c  for trans = 'T'or't','C'or'c',
     * where c is an n-by-n symmetric matrix;
     * a is an n-by-k matrix, if trans = 'N'or'n',
     * a is a k-by-n matrix, if trans = 'T'or't','C'or'c'.
     *  @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param beta
     * @param C
     */
    @Override
    public void syrk(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNumber beta,
                    IComplexNDArray C) {
        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            zsyrk(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), alpha.asDouble(), A, (int) A.size(0), beta.asDouble(), C, (int) C.size(0));
        else
            csyrk(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), alpha.asFloat(), A, (int) A.size(0), beta.asFloat(), C, (int) C.size(0));

    }

    /**
     * yr2k performs a rank-2k update of an n-by-n symmetric matrix c, that is, one of the following operations:
     * c := alpha*a*b' + alpha*b*a' + beta*c  for trans = 'N'or'n'
     * c := alpha*a'*b + alpha*b'*a + beta*c  for trans = 'T'or't',
     * where c is an n-by-n symmetric matrix;
     * a and b are n-by-k matrices, if trans = 'N'or'n',
     * a and b are k-by-n matrices, if trans = 'T'or't'.
     *  @param Order
     * @param Uplo
     * @param Trans
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     */
    @Override
    public void syr2k(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            zsyr2k(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), alpha.asDouble(), A, (int) A.size(0), B, (int) B.size(0), beta.asDouble(), C, (int) C.size(0));
        else
            csyr2k(Order, Uplo, Trans, (int) A.rows(), (int) A.columns(), alpha.asFloat(), A, (int) A.size(0), B, (int) B.size(0), beta.asFloat(), C, (int) C.size(0));

    }

    /**
     * syr2k performs a rank-2k update of an n-by-n symmetric matrix c, that is, one of the following operations:
     * c := alpha*a*b' + alpha*b*a' + beta*c  for trans = 'N'or'n'
     * c := alpha*a'*b + alpha*b'*a + beta*c  for trans = 'T'or't',
     * where c is an n-by-n symmetric matrix;
     * a and b are n-by-k matrices, if trans = 'N'or'n',
     * a and b are k-by-n matrices, if trans = 'T'or't'.
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
    @Override
    public void trmm(char Order, char Side, char Uplo, char TransA, char Diag, IComplexNumber alpha, IComplexNDArray A,
                    IComplexNDArray B, IComplexNDArray C) {
        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            ztrmm(Order, Side, Uplo, TransA, Diag, (int) A.rows(), (int) A.columns(), alpha.asDouble(), A, (int) A.size(0), B, (int) B.size(0), C, (int) C.size(0));
        else
            ctrmm(Order, Side, Uplo, TransA, Diag, (int) A.rows(), (int) A.columns(), alpha.asFloat(), A, (int) A.size(0), B, (int) B.size(0), C, (int) C.size(0));

    }

    /**
     * ?trsm solves one of the following matrix equations:
     * op(a)*x = alpha*b  or  x*op(a) = alpha*b,
     * where x and b are m-by-n general matrices, and a is triangular;
     * op(a) must be an m-by-m matrix, if side = 'L'or'l'
     * op(a) must be an n-by-n matrix, if side = 'R'or'r'.
     * For the definition of op(a), see Matrix Arguments.
     * The routine overwrites x on b.
     *  @param Order
     * @param Side
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param alpha
     * @param A
     * @param B
     */
    @Override
    public void trsm(char Order, char Side, char Uplo, char TransA, char Diag, IComplexNumber alpha, IComplexNDArray A,
                    IComplexNDArray B) {

        // FIXME: int cast

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            ztrsm(Order, Side, Uplo, TransA, Diag, (int) A.rows(), (int) A.columns(), alpha.asDouble(), A, (int) A.size(0), B, (int) B.size(0));
        else
            ctrsm(Order, Side, Uplo, TransA, Diag, (int) A.rows(), (int) A.columns(), alpha.asFloat(), A, (int) A.size(0), B, (int) B.size(0));

    }

    /*
     * ===========================================================================
     * Prototypes for level 3 BLAS
     * ===========================================================================
     */

    /* 
     * Routines with standard 4 prefixes (S, D, C, Z)
     */
    protected abstract void hgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A,
                    int lda, INDArray B, int ldb, float beta, INDArray C, int ldc);


    protected abstract void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A,
                    int lda, INDArray B, int ldb, float beta, INDArray C, int ldc);

    protected abstract void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda,
                    INDArray B, int ldb, float beta, INDArray C, int ldc);

    protected abstract void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda,
                    float beta, INDArray C, int ldc);

    protected abstract void ssyr2k(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda,
                    INDArray B, int ldb, float beta, INDArray C, int ldc);

    protected abstract void strmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha,
                    INDArray A, int lda, INDArray B, int ldb);

    protected abstract void strsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha,
                    INDArray A, int lda, INDArray B, int ldb);

    protected abstract void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A,
                    int lda, INDArray B, int ldb, double beta, INDArray C, int ldc);

    protected abstract void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda,
                    INDArray B, int ldb, double beta, INDArray C, int ldc);

    protected abstract void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda,
                    double beta, INDArray C, int ldc);

    protected abstract void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda,
                    INDArray B, int ldb, double beta, INDArray C, int ldc);

    protected abstract void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha,
                    INDArray A, int lda, INDArray B, int ldb);

    protected abstract void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha,
                    INDArray A, int lda, INDArray B, int ldb);

    protected abstract void cgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C,
                    int ldc);

    protected abstract void csymm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C,
                    int ldc);

    protected abstract void csyrk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexFloat beta, IComplexNDArray C, int ldc);

    protected abstract void csyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C,
                    int ldc);

    protected abstract void ctrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N,
                    IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C,
                    int ldc);

    protected abstract void ctrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N,
                    IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb);

    protected abstract void zgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C,
                    int ldc);

    protected abstract void zsymm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C,
                    int ldc);

    protected abstract void zsyrk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexDouble beta, IComplexNDArray C, int ldc);

    protected abstract void zsyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C,
                    int ldc);

    protected abstract void ztrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N,
                    IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C,
                    int ldc);

    protected abstract void ztrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N,
                    IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb);


    /* 
     * Routines with prefixes C and Z only
     */
    protected abstract void chemm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C,
                    int ldc);

    protected abstract void cherk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexFloat beta, IComplexNDArray C, int ldc);

    protected abstract void cher2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C,
                    int ldc);

    protected abstract void zhemm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C,
                    int ldc);

    protected abstract void zherk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexDouble beta, IComplexNDArray C, int ldc);

    protected abstract void zher2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C,
                    int ldc);

}
