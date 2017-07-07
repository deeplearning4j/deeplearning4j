package org.nd4j.linalg.jcublas.blas;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.linalg.api.blas.impl.BaseLevel2;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.Nd4jBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.javacpp.cublas.*;
import static org.bytedeco.javacpp.cuda.CUstream_st;
import static org.nd4j.linalg.jcublas.blas.CudaBlas.convertTranspose;

/**
 * @author Adam Gibson
 */
public class JcublasLevel2 extends BaseLevel2 {
    private Allocator allocator = AtomicAllocator.getInstance();
    private Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private static Logger logger = LoggerFactory.getLogger(JcublasLevel2.class);

    @Override
    protected void sgemv(char order, char TransA, int M, int N, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            logger.warn("FLOAT gemv called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(Y, A, X);

        CublasPointer cAPointer = new CublasPointer(A, ctx);
        CublasPointer cBPointer = new CublasPointer(X, ctx);
        CublasPointer cCPointer = new CublasPointer(Y, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasSgemv_v2(new cublasContext(handle), convertTranspose(TransA), M, N, new FloatPointer(alpha),
                            (FloatPointer) cAPointer.getDevicePointer(), lda,
                            (FloatPointer) cBPointer.getDevicePointer(), incX, new FloatPointer(beta),
                            (FloatPointer) cCPointer.getDevicePointer(), incY);
        }

        allocator.registerAction(ctx, Y, A, X);
        OpExecutionerUtil.checkForAny(Y);
    }

    @Override
    protected void sgbmv(char order, char TransA, int M, int N, int KL, int KU, float alpha, INDArray A, int lda,
                    INDArray X, int incX, float beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void strmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void stbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void stpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void strsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void stbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void stpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dgemv(char order, char TransA, int M, int N, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            logger.warn("DOUBLE gemv called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(Y, A, X);

        CublasPointer cAPointer = new CublasPointer(A, ctx);
        CublasPointer cBPointer = new CublasPointer(X, ctx);
        CublasPointer cCPointer = new CublasPointer(Y, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDgemv_v2(new cublasContext(handle), convertTranspose(TransA), M, N, new DoublePointer(alpha),
                            (DoublePointer) cAPointer.getDevicePointer(), lda,
                            (DoublePointer) cBPointer.getDevicePointer(), incX, new DoublePointer(beta),
                            (DoublePointer) cCPointer.getDevicePointer(), incY);
        }

        allocator.registerAction(ctx, Y, A, X);
        OpExecutionerUtil.checkForAny(Y);
    }

    @Override
    protected void dgbmv(char order, char TransA, int M, int N, int KL, int KU, double alpha, INDArray A, int lda,
                    INDArray X, int incX, double beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtrmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtrsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dtpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cgemv(char order, char TransA, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();


    }

    @Override
    protected void cgbmv(char order, char TransA, int M, int N, int KL, int KU, IComplexFloat alpha, IComplexNDArray A,
                    int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctbmv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctpmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctbsv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctpsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgemv(char order, char TransA, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();


    }

    @Override
    protected void zgbmv(char order, char TransA, int M, int N, int KL, int KU, IComplexDouble alpha, IComplexNDArray A,
                    int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztbmv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztpmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztbsv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztpsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ssymv(char order, char Uplo, int N, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ssbmv(char order, char Uplo, int N, int K, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sspmv(char order, char Uplo, int N, float alpha, INDArray Ap, INDArray X, int incX, float beta,
                    INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sger(char order, int M, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A,
                    int lda) {
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
    protected void ssyr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sspr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dsymv(char order, char Uplo, int N, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dsbmv(char order, char Uplo, int N, int K, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dspmv(char order, char Uplo, int N, double alpha, INDArray Ap, INDArray X, int incX, double beta,
                    INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dger(char order, int M, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A,
                    int lda) {
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
    protected void dsyr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void dspr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chemv(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chbmv(char order, char Uplo, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpmv(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray Ap, IComplexNDArray X,
                    int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cgeru(char order, int M, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y,
                    int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cgerc(char order, int M, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y,
                    int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void cher(char order, char Uplo, int N, float alpha, IComplexNDArray X, int incX, IComplexNDArray A,
                    int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpr(char order, char Uplo, int N, INDArray alpha, IComplexNDArray X, int incX, IComplexNDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cher2(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray X, int incX,
                    IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpr2(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray X, int incX,
                    IComplexNDArray Y, int incY, IComplexNDArray Ap) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhemv(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhbmv(char order, char Uplo, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpmv(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray Ap, IComplexNDArray X,
                    int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgeru(char order, int M, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y,
                    int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgerc(char order, int M, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y,
                    int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zher(char order, char Uplo, int N, double alpha, IComplexNDArray X, int incX, IComplexNDArray A,
                    int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpr(char order, char Uplo, int N, INDArray alpha, IComplexNDArray X, int incX, IComplexNDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zher2(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray X, int incX,
                    IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpr2(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray X, int incX,
                    IComplexNDArray Y, int incY, IComplexNDArray Ap) {
        throw new UnsupportedOperationException();

    }
}
