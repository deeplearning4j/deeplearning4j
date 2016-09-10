package org.nd4j.linalg.jcublas.blas;


import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.linalg.api.blas.impl.BaseLevel3;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.Nd4jBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Level 3 implementation of matrix matrix operations
 *
 * @author Adam Gibson
 */
public class JcublasLevel3 extends BaseLevel3 {
    private Allocator allocator = AtomicAllocator.getInstance();
    private Nd4jBlas nd4jBlas = new Nd4jBlas();
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private static Logger logger = LoggerFactory.getLogger(JcublasLevel3.class);

    @Override
    protected void hgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        //A = Shape.toOffsetZero(A);
        //B = Shape.toOffsetZero(B);

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer cAPointer = new CublasPointer(A, ctx);
        CublasPointer cBPointer = new CublasPointer(B, ctx);
        CublasPointer cCPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.hgemm(
                    new PointerPointer(new Pointer[] {ctx.getHandle()}),
                    Order,
                    TransA,
                    TransB,
                    M,
                    N,
                    K,
                    alpha,
                    (ShortPointer)cAPointer.getDevicePointer(),
                    lda,
                    (ShortPointer)cBPointer.getDevicePointer(),
                    ldb,
                    beta,
                    (ShortPointer)cCPointer.getDevicePointer(),
                    ldc);
        }

        allocator.registerAction(ctx, C, A, B);
    }


    @Override
    protected void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        //A = Shape.toOffsetZero(A);
        //B = Shape.toOffsetZero(B);
        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            logger.warn("FLOAT gemm called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer cAPointer = new CublasPointer(A, ctx);
        CublasPointer cBPointer = new CublasPointer(B, ctx);
        CublasPointer cCPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.sgemm(
                    new PointerPointer(new Pointer[] {ctx.getHandle()}),
                    Order,
                    TransA,
                    TransB,
                    M,
                    N,
                    K,
                    alpha,
                    (FloatPointer)cAPointer.getDevicePointer(),
                    lda,
                    (FloatPointer)cBPointer.getDevicePointer(),
                    ldb,
                    beta,
                    (FloatPointer)cCPointer.getDevicePointer(),
                    ldc);
        }

        allocator.registerAction(ctx, C, A, B);
    }

    @Override
    protected void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            logger.warn("FLOAT symm called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer aPointer = new CublasPointer(A, ctx);
        CublasPointer bPointer = new CublasPointer(B, ctx);
        CublasPointer cPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.ssymm(new PointerPointer(new Pointer[] {ctx.getHandle()}),
                    Order,
                    Side,
                    Uplo,
                    M, N,
                    alpha,
                    (FloatPointer)aPointer.getDevicePointer(),
                    lda, (FloatPointer)bPointer.getDevicePointer(),
                    ldb,
                    beta,
                    (FloatPointer)cPointer.getDevicePointer(),
                    ldc);
        }

        allocator.registerAction(ctx, C, A, B);
    }

    @Override
    protected void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, float beta, INDArray C, int ldc) {

        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
        logger.warn("FLOAT syrk called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A);

        CublasPointer aPointer = new CublasPointer(A,ctx);
        CublasPointer cPointer = new CublasPointer(C,ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.ssyrk(new PointerPointer(new Pointer[] {ctx.getHandle()}), Order, Uplo, Trans, N, K, alpha, (FloatPointer)aPointer.getDevicePointer(), lda, beta, (FloatPointer)cPointer.getDevicePointer(), ldc);
        }

        allocator.registerAction(ctx, C, A);
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
        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            logger.warn("FLOAT trsm called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(B, A);

        CublasPointer aPointer = new CublasPointer(A,ctx);
        CublasPointer bPointer = new CublasPointer(B,ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.strsm(new PointerPointer(new Pointer[] {ctx.getHandle()}),
                    Order,
                    Side,
                    Uplo,
                    TransA,
                    Diag,
                    M,
                    N,
                    alpha,
                    (FloatPointer)aPointer.getDevicePointer(),
                    lda,
                    (FloatPointer)bPointer.getDevicePointer(),
                    ldb);
        }

        allocator.registerAction(ctx, B, A);
    }

    @Override
    protected void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        //A = Shape.toOffsetZero(A);
        //B = Shape.toOffsetZero(B);
        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            logger.warn("DOUBLE gemm called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        DataTypeValidation.assertDouble(A, B, C);

        CublasPointer cAPointer = new CublasPointer(A,ctx);
        CublasPointer cBPointer = new CublasPointer(B,ctx);
        CublasPointer cCPointer = new CublasPointer(C,ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.dgemm(new PointerPointer(new Pointer[] {ctx.getHandle()}), Order, TransA, TransB, M, N, K, alpha, (DoublePointer)cAPointer.getDevicePointer(), lda, (DoublePointer)cBPointer.getDevicePointer(), ldb, beta, (DoublePointer)cCPointer.getDevicePointer(), ldc);
            ctx.syncOldStream();
        }

        allocator.registerAction(ctx, C, A, B);
    }

    @Override
    protected void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            logger.warn("DOUBLE symm called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer aPointer = new CublasPointer(A,ctx);
        CublasPointer bPointer = new CublasPointer(B,ctx);
        CublasPointer cPointer = new CublasPointer(C,ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.dsymm(new PointerPointer(new Pointer[] {ctx.getHandle()}),
                    Order,
                    Side,
                    Uplo,
                    M,
                    N,
                    alpha,
                    (DoublePointer)aPointer.getDevicePointer(),
                    lda,
                    (DoublePointer)bPointer.getDevicePointer(),
                    ldb,
                    beta,
                    (DoublePointer)cPointer.getDevicePointer(),
                    ldc);
        }

        allocator.registerAction(ctx, C, A, B);
    }

    @Override
    protected void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, double beta, INDArray C, int ldc) {
        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            logger.warn("DOUBLE syrk called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A);

        CublasPointer aPointer = new CublasPointer(A,ctx);
        CublasPointer cPointer = new CublasPointer(C,ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.dsyrk(new PointerPointer(new Pointer[] {ctx.getHandle()}),
                    Order,
                    Uplo,
                    Trans,
                    N,
                    K,
                    alpha,
                    (DoublePointer)aPointer.getDevicePointer(),
                    lda,
                    beta, (DoublePointer)cPointer.getDevicePointer(),
                    ldc);
        }

        allocator.registerAction(ctx, C, A);
    }

    @Override
    protected void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            logger.warn("DOUBLE syr2k called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer aPointer = new CublasPointer(A,ctx);
        CublasPointer bPointer = new CublasPointer(B,ctx);
        CublasPointer cPointer = new CublasPointer(C,ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.dsyr2k(new PointerPointer(new Pointer[] {ctx.getHandle()}),
                    Order,
                    Uplo,
                    Trans,
                    N,
                    K,
                    alpha,
                    (DoublePointer)aPointer.getDevicePointer(),
                    lda,
                    (DoublePointer)bPointer.getDevicePointer(),
                    ldb,
                    beta,
                    (DoublePointer)cPointer.getDevicePointer(),
                    ldc);
        }

        allocator.registerAction(ctx, C, A, B);
    }

    @Override
    protected void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            logger.warn("DOUBLE trmm called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(B, A);

        CublasPointer aPointer = new CublasPointer(A,ctx);
        CublasPointer bPointer = new CublasPointer(B,ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.dtrmm(new PointerPointer(new Pointer[] {ctx.getHandle()}),
                    Order,
                    Side,
                    Uplo,
                    TransA,
                    Diag,
                    M,
                    N,
                    alpha,
                    (DoublePointer)aPointer.getDevicePointer(),
                    lda,
                    (DoublePointer)bPointer.getDevicePointer(),
                    ldb);
        }

        allocator.registerAction(ctx, B, A);
    }

    @Override
    protected void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            logger.warn("DOUBLE trsm called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        CudaContext ctx = allocator.getFlowController().prepareAction(B, A);

        CublasPointer aPointer = new CublasPointer(A,ctx);
        CublasPointer bPointer = new CublasPointer(B,ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            nativeOps.setBlasStream(handle, ctx.getOldStream());

            nd4jBlas.dtrsm(new PointerPointer(new Pointer[] {ctx.getHandle()}),
                    Order,
                    Side,
                    Uplo,
                    TransA,
                    Diag,
                    M,
                    N,
                    alpha,
                    (DoublePointer)aPointer.getDevicePointer(),
                    lda,
                    (DoublePointer)bPointer.getDevicePointer(),
                    ldb);
        }

        allocator.registerAction(ctx, B, A);
    }

    @Override
    protected void cgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void csymm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void csyrk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void csyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {

    }

    @Override
    protected void zgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zsymm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

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
    protected void ztrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C, int ldc) {
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
