package org.nd4j.linalg.jcublas.blas;

import jcuda.Pointer;
import jcuda.jcublas.JCublas2;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.blas.impl.BaseLevel3;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.util.OpUtil;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.nativeblas.DefaultPointerConverter;
import org.nd4j.nativeblas.Nd4jBlas;
import org.nd4j.nativeblas.PointerConverter;

/**
 * Level 3 implementation of matrix matrix operations
 *
 * @author Adam Gibson
 */
public class JcublasLevel3 extends BaseLevel3 {
    private Allocator allocator = AtomicAllocator.getInstance();
    private Nd4jBlas nd4jBlas = new Nd4jBlas();
    private PointerConverter pointerConverter = new DefaultPointerConverter();

    @Override
    protected void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        A = Shape.toOffsetZero(A);
        B = Shape.toOffsetZero(B);
        CudaContext ctx = CudaContext.getBlasContext();


        try(CublasPointer cAPointer = new CublasPointer(A,ctx);
            CublasPointer cBPointer = new CublasPointer(B,ctx);
            CublasPointer cCPointer = new CublasPointer(C,ctx)) {

            nd4jBlas.sgemm(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()}
            ,Order,TransA,TransB,M,N,K,alpha,
                    cAPointer.getDevicePointer().getNativePointer(),
                    lda,
                    cBPointer.getDevicePointer().getNativePointer(),
                    ldb,
                    beta,
                    cCPointer.getDevicePointer().getNativePointer(),
                    ldc);

         /*   JCublas2.cublasSgemm(
                    ctx.getHandle(),
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
                    ldc);*/
            ctx.syncOldStream();
     //       cCPointer.copyToHost();

            allocator.tickDeviceWrite(C);


            allocator.tackDevice(A);
            allocator.tackDevice(B);
            allocator.tackDevice(C);


        }catch (Exception e) {
            throw new RuntimeException(e);
        }

        finally {
            ctx.finishBlasOperation();

        }
    }

    @Override
    protected void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        CudaContext ctx = CudaContext.getBlasContext();

        try(CublasPointer aPointer = new CublasPointer(A,ctx);
            CublasPointer bPointer = new CublasPointer(B,ctx);
            CublasPointer cPointer = new CublasPointer(C,ctx)) {
            nd4jBlas.ssymm(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()},
                    Order,
                    Side,
                    Uplo,
                    M,N,
                    alpha,
                    aPointer.getDevicePointer().getNativePointer(),
                    lda,bPointer.getDevicePointer().getNativePointer(),
                    ldb,
                    beta,
                    cPointer.getDevicePointer().getNativePointer(),
                    ldc);
           /* JCublas2.cublasSsymm(
                    ctx.getHandle(),
                    OpUtil.getOp(Order),
                    OpUtil.getOp(Uplo),
                    M,
                    N,
                    PointerUtil.getPointer(alpha),
                    aPointer.getDevicePointer()
                    , lda,
                    bPointer.getDevicePointer()
                    , ldb,
                    PointerUtil.getPointer(beta)
                    , cPointer.getDevicePointer()
                    , ldc);*/
            ctx.syncOldStream();
        //    cPointer.copyToHost();

            allocator.tickDeviceWrite(C);


            allocator.tackDevice(A);
            allocator.tackDevice(B);
            allocator.tackDevice(C);

        }catch (Exception e) {
            throw new RuntimeException(e);
        }
        finally {
            ctx.finishBlasOperation();

        }
    }

    @Override
    protected void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, float beta, INDArray C, int ldc) {
        CudaContext ctx = CudaContext.getBlasContext();
        try(CublasPointer aPointer = new CublasPointer(A,ctx);
            CublasPointer cPointer = new CublasPointer(C,ctx)) {
            nd4jBlas.ssyrk(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()},Order,Uplo,Trans,N,K,alpha,aPointer.getDevicePointer().getNativePointer(),lda,beta,cPointer.getDevicePointer().getNativePointer(),ldc);
           // JCublas2.cublasSsyrk(ctx.getHandle(),OpUtil.getOp(Order),OpUtil.getOp(Trans),N,K,PointerUtil.getPointer(alpha),aPointer.getDevicePointer(),lda,PointerUtil.getPointer(beta),cPointer.getDevicePointer(),ldc);
            ctx.syncOldStream();
        //    cPointer.copyToHost();

            allocator.tickDeviceWrite(C);


            allocator.tackDevice(A);
            allocator.tackDevice(C);

        }catch (Exception e) {
            throw new RuntimeException(e);
        }
        finally {
            ctx.finishBlasOperation();
        }
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
        CudaContext ctx = CudaContext.getBlasContext();
        try(CublasPointer aPointer = new CublasPointer(A,ctx);
            CublasPointer bPointer = new CublasPointer(B,ctx)) {
            nd4jBlas.strsm(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()},Order,Side,Uplo,TransA,Diag,M,N,alpha,aPointer.getDevicePointer().getNativePointer(),lda,bPointer.getDevicePointer().getNativePointer(),ldb);
          /*  JCublas2.cublasStrsm(ctx.getHandle()
                    ,OpUtil.getOp(Side),OpUtil.getOp(Uplo)
                    ,OpUtil.getOp(TransA),OpUtil.getOp(Diag)
                    ,M,
                    N
                    ,PointerUtil.getPointer(alpha)
                    ,aPointer.getDevicePointer()
                    ,lda
                    ,bPointer.getDevicePointer()
                    ,ldb);*/
            ctx.syncOldStream();
        //    bPointer.copyToHost();

            allocator.tickDeviceWrite(B);


            allocator.tackDevice(A);
            allocator.tackDevice(B);

        }catch (Exception e) {
            throw new RuntimeException(e);
        }
        finally {
            ctx.finishBlasOperation();

        }
    }

    @Override
    protected void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        A = Shape.toOffsetZero(A);
        B = Shape.toOffsetZero(B);
        CudaContext ctx = CudaContext.getBlasContext();


        DataTypeValidation.assertDouble(A, B, C);



        try(CublasPointer cAPointer = new CublasPointer(A,ctx);
            CublasPointer cBPointer = new CublasPointer(B,ctx);
            CublasPointer cCPointer = new CublasPointer(C,ctx)) {
          nd4jBlas.dgemm(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()},Order,TransA,TransB,M,N,K,alpha,cAPointer.getDevicePointer().getNativePointer(),lda,cBPointer.getDevicePointer().getNativePointer(),ldb,beta,cCPointer.getDevicePointer().getNativePointer(),ldc);
            /*JCublas2.cublasDgemm(
                  ctx.getHandle(),
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
                    ldc); // incy*/
            ctx.syncOldStream();
        //    cCPointer.copyToHost();

            allocator.tickDeviceWrite(C);

            allocator.tackDevice(A);
            allocator.tackDevice(B);
            allocator.tackDevice(C);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
        finally {
            ctx.finishBlasOperation();
        }

    }

    @Override
    protected void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        CudaContext ctx = CudaContext.getBlasContext();

        try(CublasPointer aPointer = new CublasPointer(A,ctx);
            CublasPointer bPointer = new CublasPointer(B,ctx);
            CublasPointer cPointer = new CublasPointer(C,ctx)) {
            nd4jBlas.dsymm(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()},
                    Order,
                    Side,
                    Uplo,
                    M,
                    N,
                    alpha,
                    aPointer.getDevicePointer().getNativePointer(),
                    lda,
                    bPointer.getDevicePointer().getNativePointer(),
                    ldb,
                    beta,
                    cPointer.getDevicePointer().getNativePointer(),
                    ldc);
            ctx.syncOldStream();

            allocator.tickDeviceWrite(C);

            allocator.tackDevice(A);
            allocator.tackDevice(B);
            allocator.tackDevice(C);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        finally {
            ctx.finishBlasOperation();
        }

    }

    @Override
    protected void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, double beta, INDArray C, int ldc) {
        CudaContext ctx = CudaContext.getBlasContext();
        try(CublasPointer aPointer = new CublasPointer(A,ctx);
            CublasPointer cPointer = new CublasPointer(C,ctx)) {
            nd4jBlas.dsyrk(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()},
                    Order,
                    Uplo,
                    Trans,
                    N,
                    K,
                    alpha,
                    aPointer.getDevicePointer().getNativePointer(),
                    lda,
                    beta,cPointer.getDevicePointer().getNativePointer(),
                    ldc);
            ctx.syncOldStream();

            allocator.tickDeviceWrite(C);

            allocator.tackDevice(A);
            allocator.tackDevice(C);

        }catch(Exception e) {
            throw new RuntimeException(e);
        }
        finally {
            ctx.finishBlasOperation();

        }

    }

    @Override
    protected void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        CudaContext ctx = CudaContext.getBlasContext();
        try(CublasPointer aPointer = new CublasPointer(A,ctx);
            CublasPointer bPointer = new CublasPointer(B,ctx);
            CublasPointer cPointer = new CublasPointer(C,ctx)) {
            nd4jBlas.dsyr2k(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()},
                    Order,
                    Uplo,
                    Trans,
                    N,
                    K,
                    alpha,
                    aPointer.getDevicePointer().getNativePointer(),
                    lda,
                    bPointer.getDevicePointer().getNativePointer(),
                    ldb,
                    beta,
                    cPointer.getDevicePointer().getNativePointer(),
                    ldc);
            ctx.syncOldStream();

            allocator.tickDeviceWrite(C);

            allocator.tackDevice(A);
            allocator.tackDevice(B);
            allocator.tackDevice(C);
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
        finally {
            ctx.finishBlasOperation();
        }

    }

    @Override
    protected void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        CudaContext ctx = CudaContext.getBlasContext();
        try(CublasPointer aPointer = new CublasPointer(A,ctx);
            CublasPointer bPointer = new CublasPointer(B,ctx)) {
            nd4jBlas.dtrmm(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()},
                    Order,
                    Side,
                    Uplo,
                    TransA,
                    Diag,
                    M,
                    N,
                    alpha,
                    aPointer.getDevicePointer().getNativePointer(),
                    lda,
                    bPointer.getDevicePointer().getNativePointer(),
                    ldb);

            ctx.syncOldStream();

            allocator.tickDeviceWrite(B);
            allocator.tackDevice(A);
            allocator.tackDevice(B);

        }catch (Exception e) {
            throw new RuntimeException(e);
        }
        finally {
            ctx.finishBlasOperation();
        }

    }

    @Override
    protected void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        CudaContext ctx = CudaContext.getBlasContext();

        try(CublasPointer aPointer = new CublasPointer(A,ctx);
            CublasPointer bPointer = new CublasPointer(B,ctx)) {
            nd4jBlas.dtrsm(new long[]{pointerConverter.toPointer(A.shapeInfo()),ctx.getHandle().getNativePointer()},
                    Order,
                    Side,
                    Uplo,
                    TransA,
                    Diag,
                    M,
                    N,
                    alpha,
                    aPointer.getDevicePointer().getNativePointer(),
                    lda,
                    bPointer.getDevicePointer().getNativePointer(),
                    ldb);

            ctx.syncOldStream();

            allocator.tickDeviceWrite(B);
            allocator.tackDevice(A);
            allocator.tackDevice(B);

        }catch (Exception e) {
            throw new RuntimeException(e);
        }
        finally {
            ctx.finishBlasOperation();
        }

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
