package jcublas.blas;

import jcuda.Pointer;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasOperation;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.blas.impl.BaseLevel2;
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
 * @author Adam Gibson
 */
public class JcublasLevel2 extends BaseLevel2 {
    @Override
    protected void sgemv(char order, char TransA, int M, int N, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        SimpleJCublas.sync();

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(X);
        CublasPointer cCPointer = new CublasPointer(Y);

        JCublas2.cublasSgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N,
                M,
                N,
                Pointer.to(new float[]{alpha}),
                cAPointer.getDevicePointer(),
                lda,
                cBPointer.getDevicePointer(),
                incX,
                Pointer.to(new float[]{beta}),
                cCPointer.getDevicePointer(),
                incY);
        SimpleJCublas.sync();

        cCPointer.copyToHost();
    }

    @Override
    protected void sgbmv(char order, char TransA, int M, int N, int KL, int KU, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void strmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void stbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void stpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void strsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void stbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void stpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dgemv(char order, char TransA, int M, int N, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        SimpleJCublas.sync();


        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(X);
        CublasPointer cCPointer = new CublasPointer(Y);

        JCublas2.cublasDgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N,
                M,
                N,
                Pointer.to(new double[]{alpha}),
                cAPointer.getDevicePointer(),
                lda,
                cBPointer.getDevicePointer(),
                incX,
                Pointer.to(new double[]{beta}),
                cCPointer.getDevicePointer(),
                incY);

        cCPointer.copyToHost();
        SimpleJCublas.sync();
    }

    @Override
    protected void dgbmv(char order, char TransA, int M, int N, int KL, int KU, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtrmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtrsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cgemv(char order, char TransA, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        SimpleJCublas.sync();

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(X);
        CublasPointer cCPointer = new CublasPointer(Y);


        cuComplex alpha2 = cuComplex.cuCmplx(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue());
        cuComplex beta2 = cuComplex.cuCmplx(beta.realComponent().floatValue(), beta.imaginaryComponent().floatValue());

        JCublas2.cublasCgemv(
                ContextHolder.getInstance().getHandle(),
                OpUtil.getOp(BlasBufferUtil.getCharForTranspose(A)),
                M,  // m
                N, // n
                PointerUtil.getPointer(alpha2),
                cAPointer.getDevicePointer(), // A
                lda,  // lda
                cBPointer.getDevicePointer(), // x
                incX, // ldb
                PointerUtil.getPointer(beta2),  // beta
                cCPointer.getDevicePointer(), // y
                incY); // ldc

        SimpleJCublas.sync();

        cCPointer.copyToHost();


    }

    @Override
    protected void cgbmv(char order, char TransA, int M, int N, int KL, int KU, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctbmv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctpmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctbsv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctpsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgemv(char order, char TransA, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        SimpleJCublas.sync();

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(X);
        CublasPointer cCPointer = new CublasPointer(Y);

        A = (IComplexNDArray) Shape.toOffsetZero(A);
        X = (IComplexNDArray) Shape.toOffsetZero(X);

        cuDoubleComplex alpha2 = cuDoubleComplex.cuCmplx(alpha.realComponent().doubleValue(), alpha.imaginaryComponent().doubleValue());
        cuDoubleComplex beta2 = cuDoubleComplex.cuCmplx(beta.realComponent().doubleValue(), beta.imaginaryComponent().doubleValue());

        JCublas2.cublasZgemv(
                ContextHolder.getInstance().getHandle(),
                OpUtil.getOp(BlasBufferUtil.getCharForTranspose(A)),
                M,  // m
                N, // n
                PointerUtil.getPointer(alpha2),
                cAPointer.getDevicePointer(), // A
                lda,  // lda
                cBPointer.getDevicePointer(), // x
                incX, // ldb
                PointerUtil.getPointer(beta2),  // beta
                cCPointer.getDevicePointer(), // ydoin
                incY); // ldc

        SimpleJCublas.sync();

        cCPointer.copyToHost();


    }

    @Override
    protected void zgbmv(char order, char TransA, int M, int N, int KL, int KU, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztbmv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztpmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztbsv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztpsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ssymv(char order, char Uplo, int N, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ssbmv(char order, char Uplo, int N, int K, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sspmv(char order, char Uplo, int N, float alpha, INDArray Ap, INDArray X, int incX, float beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sger(char order, int M, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ssyr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sspr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Ap) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ssyr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sspr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dsymv(char order, char Uplo, int N, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dsbmv(char order, char Uplo, int N, int K, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dspmv(char order, char Uplo, int N, double alpha, INDArray Ap, INDArray X, int incX, double beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dger(char order, int M, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dsyr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dspr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Ap) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dsyr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dspr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chemv(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chbmv(char order, char Uplo, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpmv(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray Ap, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cgeru(char order, int M, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cgerc(char order, int M, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {

        SimpleJCublas.sync();

        A = (IComplexNDArray) Shape.toOffsetZero(A);
        X = (IComplexNDArray) Shape.toOffsetZero(X);

        CublasPointer xPointer = new CublasPointer(A);
        CublasPointer yPointer = new CublasPointer(X);
        CublasPointer aPointer = new CublasPointer(Y);


        cuComplex alpha2 = cuComplex.cuCmplx(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue());


        JCublas2.cublasCgerc(
                ContextHolder.getInstance().getHandle(),
                M,   // m
                N,// n
                PointerUtil.getPointer(alpha2),      // alpha
                xPointer.getDevicePointer(),        // dA or x
                incX,   // incx
                yPointer.getDevicePointer(),        // dB or y
                incY,   // incy
                aPointer.getDevicePointer(),        // dC or A
                lda    // lda
        );

        SimpleJCublas.sync();
        aPointer.copyToHost();
    }

    @Override
    protected void cher(char order, char Uplo, int N, float alpha, IComplexNDArray X, int incX, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpr(char order, char Uplo, int N, INDArray alpha, IComplexNDArray X, int incX, IComplexNDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cher2(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpr2(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray Ap) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhemv(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhbmv(char order, char Uplo, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpmv(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray Ap, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgeru(char order, int M, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        SimpleJCublas.sync();
        DataTypeValidation.assertDouble(A, X, Y);

        A = (IComplexNDArray) Shape.toOffsetZero(A);
        X = (IComplexNDArray) Shape.toOffsetZero(X);


        CublasPointer aCPointer = new CublasPointer(A);
        CublasPointer bCPointer = new CublasPointer(X);
        CublasPointer cCPointer = new CublasPointer(Y);

        cuDoubleComplex alpha2 = cuDoubleComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent());

        JCublas2.cublasZgeru(
                ContextHolder.getInstance().getHandle(),
                M,   // m
                N,// n
                PointerUtil.getPointer(alpha2),      // alpha
                aCPointer.getDevicePointer(),        // d_A or x
                incX,   // incx
                bCPointer.getDevicePointer(),        // d_B or y
                incY,   // incy
                cCPointer.getDevicePointer(),        // d_C or A
                lda    // lda
        );

        SimpleJCublas.sync();
        cCPointer.copyToHost();

    }

    @Override
    protected void zgerc(char order, int M, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        SimpleJCublas.sync();
        A = (IComplexNDArray) Shape.toOffsetZero(A);
        X = (IComplexNDArray) Shape.toOffsetZero(X);


        CublasPointer xPointer = new CublasPointer(A);
        CublasPointer yPointer = new CublasPointer(X);
        CublasPointer aPointer = new CublasPointer(Y);


        cuComplex alpha2 = cuComplex.cuCmplx(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue());


        JCublas2.cublasZgerc(
                ContextHolder.getInstance().getHandle(),
                M,   // m
                N,// n
                PointerUtil.getPointer(alpha2),      // alpha
                xPointer.getDevicePointer(),        // dA or x
                incX,   // incx
                yPointer.getDevicePointer(),        // dB or y
                incY,   // incy
                aPointer.getDevicePointer(),        // dC or A
                lda    // lda
        );

        SimpleJCublas.sync();

        aPointer.copyToHost();

    }

    @Override
    protected void zher(char order, char Uplo, int N, double alpha, IComplexNDArray X, int incX, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpr(char order, char Uplo, int N, INDArray alpha, IComplexNDArray X, int incX, IComplexNDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zher2(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpr2(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray Ap) {
        throw new UnsupportedOperationException();

    }
}
