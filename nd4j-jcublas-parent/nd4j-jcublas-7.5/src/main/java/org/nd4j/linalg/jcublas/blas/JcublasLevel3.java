package org.nd4j.linalg.jcublas.blas;

import jcuda.Pointer;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasOperation;
import org.nd4j.linalg.api.blas.impl.BaseLevel3;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.util.OpUtil;
import org.nd4j.linalg.jcublas.util.PointerUtil;

/**
 * Level 3 implementation of matrix matrix operations
 *
 * @author Adam Gibson
 */
public class JcublasLevel3 extends BaseLevel3 {
    @Override
    protected void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        A = Shape.toOffsetZero(A);
        B = Shape.toOffsetZero(B);


        SimpleJCublas.sync();
        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


        JCublas2.cublasSgemm(
                ContextHolder.getInstance().getHandle(),
                OpUtil.getOp(TransA),
                OpUtil.getOp(TransB),
                M,
                N,
                K,
                Pointer.to(new float[]{alpha}),
                cAPointer.getDevicePointer(),
                lda,
                cBPointer.getDevicePointer(),
                ldb,
                Pointer.to(new float[]{beta}),
                cCPointer.getDevicePointer(),
                ldc);


        cCPointer.copyToHost();
        SimpleJCublas.sync();

    }

    @Override
    protected void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasSsymm(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Uplo), M, N, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();
    }

    @Override
    protected void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, float beta, INDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasSsyrk(ContextHolder.getInstance().getHandle(),OpUtil.getOp(Order),OpUtil.getOp(Trans),N,K,PointerUtil.getPointer(alpha),aPointer.getDevicePointer(),lda,PointerUtil.getPointer(beta),cPointer.getDevicePointer(),ldc);
        cPointer.copyToHost();


    }

    @Override
    protected void ssyr2k(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void strmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb) {
        throw new UnsupportedOperationException();}

    @Override
    protected void strsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        JCublas2.cublasStrsm(ContextHolder.getInstance().getHandle(),OpUtil.getOp(Side),OpUtil.getOp(Uplo),OpUtil.getOp(TransA),OpUtil.getOp(Diag),M,N,PointerUtil.getPointer(alpha),aPointer.getDevicePointer(),lda,bPointer.getDevicePointer(),ldb);
        bPointer.copyToHost();
    }

    @Override
    protected void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        A = Shape.toOffsetZero(A);
        B = Shape.toOffsetZero(B);


        DataTypeValidation.assertDouble(A, B, C);

        SimpleJCublas.sync();


        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


        JCublas2.cublasDgemm(
                ContextHolder.getInstance().getHandle(),
                OpUtil.getOp(TransA),
                OpUtil.getOp(TransB),
                M,  // m
                N, // n
                K, //k,
                Pointer.to(new double[]{alpha}),
                cAPointer.getDevicePointer(), // A
                lda,  // lda
                cBPointer.getDevicePointer(), // x
                ldb, // ldb
                Pointer.to(new double[]{beta}),
                cCPointer.getDevicePointer(), // y
                ldc); // incy

        cCPointer.copyToHost();
        SimpleJCublas.sync();


    }

    @Override
    protected void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasDsymm(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Uplo), M, N, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();

    }

    @Override
    protected void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, double beta, INDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasDsyrk(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Trans), N, K, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();

    }

    @Override
    protected void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasDsyr2k(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Uplo), N, K, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();

    }

    @Override
    protected void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        JCublas2.cublasDtrsm(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Side), OpUtil.getOp(Uplo), OpUtil.getOp(TransA), OpUtil.getOp(Diag), M, N, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb);
        bPointer.copyToHost();

    }

    @Override
    protected void cgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        cuComplex alpha2 = cuComplex.cuCmplx(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue());
        cuComplex beta2 = cuComplex.cuCmplx(beta.realComponent().floatValue(), beta.imaginaryComponent().floatValue());
        //custom striding for blas doesn't work

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


        JCublas2.cublasCgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                M,  // m
                N, // n
                K, //k,
                PointerUtil.getPointer(alpha2),
                cAPointer.getDevicePointer(), // A
                lda,  // lda
                cBPointer.getDevicePointer(), // x
                ldb, // ldb
                PointerUtil.getPointer(beta2),  // beta
                cCPointer.getDevicePointer(), // y
                ldc); // ldc

        SimpleJCublas.sync();

        cCPointer.copyToHost();
    }

    @Override
    protected void csymm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasCsymm(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Uplo), M, N, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();
    }

    @Override
    protected void csyrk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexFloat beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasCsyrk(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Trans), N, K, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();
    }

    @Override
    protected void csyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasCsyr2k(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Uplo), N, K, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();
    }

    @Override
    protected void ctrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);

        JCublas2.cublasCtrmm(ContextHolder.getInstance().getHandle(),OpUtil.getOp(Side),OpUtil.getOp(Uplo),OpUtil.getOp(TransA),OpUtil.getOp(Diag),M,N,PointerUtil.getPointer(alpha),aPointer.getDevicePointer(),lda,bPointer.getDevicePointer(),ldb,cPointer.getDevicePointer(),ldc);
        cPointer.copyToHost();
    }

    @Override
    protected void ctrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        JCublas2.cublasCtrsm(ContextHolder.getInstance().getHandle(),OpUtil.getOp(Side),OpUtil.getOp(Uplo),OpUtil.getOp(TransA),OpUtil.getOp(Diag),M,N,PointerUtil.getPointer(alpha),aPointer.getDevicePointer(),lda,bPointer.getDevicePointer(),ldb);
        bPointer.copyToHost();
    }

    @Override
    protected void zgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        SimpleJCublas.sync();

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);



        cuDoubleComplex alpha2 = cuDoubleComplex.cuCmplx(alpha.realComponent().doubleValue(), alpha.imaginaryComponent().doubleValue());
        cuDoubleComplex beta2 = cuDoubleComplex.cuCmplx(beta.realComponent().doubleValue(), beta.imaginaryComponent().doubleValue());

        JCublas2.cublasZgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                M,  // m
                N, // n
                K, //k,
                PointerUtil.getPointer(alpha2),
                cAPointer.getDevicePointer(), // A
                lda,  // lda
                cBPointer.getDevicePointer(), // x
                ldb, // ldb
                PointerUtil.getPointer(beta2),  // beta
                cCPointer.getDevicePointer(), // y
                ldc); // ldc

        SimpleJCublas.sync();

        cCPointer.copyToHost();

    }

    @Override
    protected void zsymm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasZsymm(ContextHolder.getInstance().getHandle(),OpUtil.getOp(Order),OpUtil.getOp(Uplo),M,N,PointerUtil.getPointer(alpha),aPointer.getDevicePointer(),lda,bPointer.getDevicePointer(),ldb,PointerUtil.getPointer(beta),cPointer.getDevicePointer(),ldc);
        cPointer.copyToHost();
    }

    @Override
    protected void zsyrk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexDouble beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasZsyrk(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Trans), N, K, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();
    }

    @Override
    protected void zsyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasZsyr2k(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Uplo), N, K, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();

    }

    @Override
    protected void ztrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);

        JCublas2.cublasCtrmm(ContextHolder.getInstance().getHandle(),OpUtil.getOp(Side),OpUtil.getOp(Uplo),OpUtil.getOp(TransA),OpUtil.getOp(Diag),M,N,PointerUtil.getPointer(alpha),aPointer.getDevicePointer(),lda,bPointer.getDevicePointer(),ldb,cPointer.getDevicePointer(),ldc);
        cPointer.copyToHost();

    }

    @Override
    protected void ztrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        JCublas2.cublasZtrsm(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Side), OpUtil.getOp(Uplo), OpUtil.getOp(TransA), OpUtil.getOp(Diag), M, N, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb);
        bPointer.copyToHost();

    }

    @Override
    protected void chemm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasChemm(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Uplo), M, N, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();

    }

    @Override
    protected void cherk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexFloat beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer cPointer = new CublasPointer(C);

        JCublas2.cublasCherk(
                ContextHolder.getInstance().getHandle(),
                OpUtil.getOp(Uplo), OpUtil.getOp(Trans)
                , N
                , K
                , PointerUtil.getPointer(alpha)
                , aPointer.getDevicePointer()
                , lda
                , PointerUtil.getPointer(beta)
                , cPointer.getDevicePointer(),
                ldc);

        cPointer.copyToHost();

    }

    @Override
    protected void cher2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasCher2k(ContextHolder.getInstance().getHandle(), OpUtil.getOp(Order), OpUtil.getOp(Uplo), N, K, PointerUtil.getPointer(alpha), aPointer.getDevicePointer(), lda, bPointer.getDevicePointer(), ldb, PointerUtil.getPointer(beta), cPointer.getDevicePointer(), ldc);
        cPointer.copyToHost();


    }

    @Override
    protected void zhemm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);
        JCublas2.cublasZhemm(ContextHolder.getInstance().getHandle(),OpUtil.getOp(Order),OpUtil.getOp(Uplo),M,N,PointerUtil.getPointer(alpha),aPointer.getDevicePointer(),lda,bPointer.getDevicePointer(),ldb,PointerUtil.getPointer(beta),cPointer.getDevicePointer(),ldc);
        cPointer.copyToHost();


    }

    @Override
    protected void zherk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexDouble beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer cPointer = new CublasPointer(C);

        JCublas2.cublasZherk(
                ContextHolder.getInstance().getHandle(),
                OpUtil.getOp(Uplo), OpUtil.getOp(Trans)
                , N
                , K
                , PointerUtil.getPointer(alpha)
                , aPointer.getDevicePointer()
                , lda
                , PointerUtil.getPointer(beta)
                , cPointer.getDevicePointer(),
                ldc);

        cPointer.copyToHost();

    }

    @Override
    protected void zher2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        CublasPointer aPointer = new CublasPointer(A);
        CublasPointer bPointer = new CublasPointer(B);
        CublasPointer cPointer = new CublasPointer(C);

        JCublas2.cublasZher2k(
                ContextHolder.getInstance().getHandle(),
                OpUtil.getOp(Uplo),OpUtil.getOp(Trans)
                ,N
                ,K
                ,PointerUtil.getPointer(alpha)
                ,aPointer.getDevicePointer()
                ,lda
                ,bPointer.getDevicePointer()
                ,ldb
                ,PointerUtil.getPointer(beta)
                ,cPointer.getDevicePointer(),
                ldc);

        cPointer.copyToHost();

    }
}
