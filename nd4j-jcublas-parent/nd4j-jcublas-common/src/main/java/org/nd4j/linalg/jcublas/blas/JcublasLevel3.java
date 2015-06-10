package org.nd4j.linalg.jcublas.blas;

import jcuda.Pointer;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasOperation;
import lombok.Cleanup;
import org.nd4j.linalg.api.blas.impl.BaseLevel3;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.util.PointerUtil;

/**
 * @author Adam Gibson
 */
public class JcublasLevel3 extends BaseLevel3 {
    @Override
    protected void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        int m = A.rows();
        int n = B.columns();
        int k = A.columns();
        if(A.offset() > 0) {
            INDArray copy = Nd4j.create(A.shape());
            copy.assign(A);
            A = copy;
        }
        if(B.offset() > 0) {
            INDArray copy = Nd4j.create(B.shape());
            copy.assign(B);
            B = copy;
        }


        DataTypeValidation.assertDouble(A, B, C);

        SimpleJCublas.sync();


        @Cleanup CublasPointer cAPointer = new CublasPointer(A);
        @Cleanup CublasPointer cBPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);


        JCublas2.cublasDgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N,
                cublasOperation.CUBLAS_OP_N,
                m,
                n,
                k,
                Pointer.to(new double[]{alpha}),
                cAPointer.getDevicePointer(),
                lda,  // lda
                cBPointer.getDevicePointer(),
                ldb, // ldb
                Pointer.to(new double[]{beta}),
                cCPointer.getDevicePointer(),
                ldc);

        SimpleJCublas.sync();

        cCPointer.copyToHost();

    }

    @Override
    protected void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
             throw new UnsupportedOperationException();
    }

    @Override
    protected void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, float beta, INDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ssyr2k(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void strmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void strsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        int m = A.rows();
        int n = B.columns();
        int k = A.columns();
        if(A.offset() > 0) {
            INDArray copy = Nd4j.create(A.shape());
            copy.assign(A);
            A = copy;
        }
        if(B.offset() > 0) {
            INDArray copy = Nd4j.create(B.shape());
            copy.assign(B);
            B = copy;
        }


        DataTypeValidation.assertDouble(A, B, C);

        SimpleJCublas.sync();


        @Cleanup CublasPointer cAPointer = new CublasPointer(A);
        @Cleanup CublasPointer cBPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);


        JCublas2.cublasDgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                m,  // m
                n, // n
                k, //k,
                Pointer.to(new double[]{alpha}),
                cAPointer.getDevicePointer(), // A
                lda,  // lda
                cBPointer.getDevicePointer(), // x
                ldb, // ldb
                Pointer.to(new double[]{beta}),
                cCPointer.getDevicePointer(), // y
                ldc); // incy

        SimpleJCublas.sync();

        cCPointer.copyToHost();

    }

    @Override
    protected void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, double beta, INDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        cuComplex alpha2 = cuComplex.cuCmplx(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue());
        cuComplex beta2 = cuComplex.cuCmplx(beta.realComponent().floatValue(), beta.imaginaryComponent().floatValue());
        //custom striding for blas doesn't work

        @Cleanup CublasPointer cAPointer = new CublasPointer(A.ravel());
        @Cleanup CublasPointer cBPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);


        JCublas2.cublasCgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                C.rows(),  // m
                C.columns(), // n
                A.columns(), //k,
                PointerUtil.getPointer(alpha2),
                cAPointer.getDevicePointer(), // A
                A.rows(),  // lda
                cBPointer.getDevicePointer(), // x
                B.rows(), // ldb
                PointerUtil.getPointer(beta2),  // beta
                cCPointer.getDevicePointer(), // y
                C.rows()); // ldc

        SimpleJCublas.sync();

        cCPointer.copyToHost();
    }

    @Override
    protected void csymm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {

    }

    @Override
    protected void csyrk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexFloat beta, IComplexNDArray C, int ldc) {

    }

    @Override
    protected void csyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {

    }

    @Override
    protected void ctrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {

    }

    @Override
    protected void ctrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {

    }

    @Override
    protected void zgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        SimpleJCublas.sync();

        @Cleanup CublasPointer cAPointer = new CublasPointer(A.offset() > 0 ? A.ravel() : A);
        @Cleanup CublasPointer cBPointer = new CublasPointer(B.offset() > 0 ? B.ravel() : B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);



        cuDoubleComplex alpha2 = cuDoubleComplex.cuCmplx(alpha.realComponent().doubleValue(), alpha.imaginaryComponent().doubleValue());
        cuDoubleComplex beta2 = cuDoubleComplex.cuCmplx(beta.realComponent().doubleValue(), beta.imaginaryComponent().doubleValue());

        JCublas2.cublasZgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                C.rows(),  // m
                C.columns(), // n
                A.columns(), //k,
                PointerUtil.getPointer(alpha2),
                cAPointer.getDevicePointer(), // A
                A.size(0),  // lda
                cBPointer.getDevicePointer(), // x
                B.size(0), // ldb
                PointerUtil.getPointer(beta2),  // beta
                cCPointer.getDevicePointer(), // y
                C.size(0)); // ldc

        SimpleJCublas.sync();

        cCPointer.copyToHost();

    }

    @Override
    protected void zsymm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {

    }

    @Override
    protected void zsyrk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zsyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chemm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cherk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cher2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhemm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zherk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zher2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }
}
