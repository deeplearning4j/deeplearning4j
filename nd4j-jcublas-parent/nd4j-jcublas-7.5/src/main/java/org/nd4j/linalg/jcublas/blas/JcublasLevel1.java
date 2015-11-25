package org.nd4j.linalg.jcublas.blas;

import jcuda.Pointer;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.util.PointerUtil;

/**
 * @author Adam Gibson
 */
public class JcublasLevel1 extends BaseLevel1 {
    @Override
    protected float sdsdot(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected double dsdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }



    @Override
    protected float sdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        DataTypeValidation.assertSameDataType(X, Y);
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        CublasPointer yCPointer = new CublasPointer(Y,ctx);

        Pointer result;
        float[] ret = new float[1];
        result = Pointer.to(ret);
        JCublas2.cublasSdot(
                ctx.getHandle(),
                N,
                xCPointer.getDevicePointer(),
                incX
                , yCPointer.getDevicePointer(),
                incY, result);

        ctx.finishBlasOperation();
        try {
            xCPointer.close();
            yCPointer.close();
        }catch(Exception e) {
            throw new RuntimeException(e);
        }

        return ret[0];
    }

    @Override
    protected float sdot( int N, DataBuffer X, int offsetX, int incX, DataBuffer Y,  int offsetY, int incY){
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected double ddot(int N, INDArray X, int incX, INDArray Y, int incY) {
        double[] ret = new double[1];
        Pointer result = Pointer.to(ret);
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xCPointer = new CublasPointer(X,ctx);
        CublasPointer yCPointer = new CublasPointer(Y,ctx);

        JCublas2.cublasDdot(
                ctx.getHandle(),
                N,
                xCPointer.getDevicePointer(),
                incX
                , yCPointer.getDevicePointer(),
                incY, result);
        try {
            xCPointer.close();
            yCPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return ret[0];
    }

    @Override
    protected double ddot( int N, DataBuffer X, int offsetX, int incX, DataBuffer Y,  int offsetY, int incY){
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void cdotu_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotu) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void cdotc_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zdotu_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotu) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zdotc_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected float snrm2(int N, INDArray X, int incX) {

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer cAPointer = new CublasPointer(X,ctx);

        float[] ret = new float[1];
        Pointer result = Pointer.to(ret);
        JCublas2.cublasSnrm2(
                ctx.getHandle()
                ,N
                ,cAPointer.getDevicePointer(),
                incX
                , result);
        try {
            cAPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return ret[0];
    }

    @Override
    protected float sasum(int N, INDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        float[] ret = new float[1];
        Pointer result = Pointer.to(ret);
        JCublas2.cublasSasum(ctx.getHandle(), N, xCPointer.getDevicePointer(), incX, result);
        try {
            xCPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
           throw new RuntimeException(e);
        }
        return ret[0];
    }

    @Override
    protected float sasum(int N, DataBuffer X, int offsetX, int incX){
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incX) {
        double[] ret = new double[1];
        Pointer result = Pointer.to(ret);
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer cAPointer = new CublasPointer(X,ctx);

        JCublas2.cublasDnrm2(
                ctx.getHandle()
                , N,
                cAPointer.getDevicePointer()
                , incX
                , result);
        try {
            cAPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
           throw new RuntimeException(e);
        }
        return ret[0];
    }

    @Override
    protected double dasum(int N, INDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        float[] ret = new float[1];
        Pointer result = Pointer.to(ret);
        JCublas2.cublasDasum(
                ctx.getHandle()
                , N,
                xCPointer.getDevicePointer(),
                incX,
                result);
        try {

            xCPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return ret[0];
    }

    @Override
    protected double dasum(int N, DataBuffer X, int offsetX, int incX){
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected float scnrm2(int N, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        float[] ret = new float[1];
        Pointer result = Pointer.to(ret);
        JCublas2.cublasScnrm2(ContextHolder.getInstance().getHandle(), N, xCPointer.getDevicePointer(), incX, result);
        try {
            xCPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return ret[0];
    }

    @Override
    protected float scasum(int N, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        float[] ret = new float[1];
        Pointer result = Pointer.to(ret);
        JCublas2.cublasScasum(
                ctx.getHandle()
                , N
                , xCPointer.getDevicePointer()
                , incX
                , result);
        try {
            xCPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return ret[0];
    }

    @Override
    protected double dznrm2(int N, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        double[] ret = new double[1];
        Pointer result = Pointer.to(ret);
        JCublas2.cublasDznrm2(ContextHolder.getInstance().getHandle(), N, xCPointer.getDevicePointer(), incX, result);
        try {
            xCPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return ret[0];
    }

    @Override
    protected double dzasum(int N, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        double[] ret = new double[1];
        Pointer result = Pointer.to(ret);
        JCublas2.cublasDzasum(ContextHolder.getInstance().getHandle(), N, xCPointer.getDevicePointer(), incX, result);
        try {
            xCPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return ret[0];
    }

    @Override
    protected int isamax(int N, INDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);

        int ret2 = JCublas.cublasIsamax(
                N,
                xCPointer.getDevicePointer(),
                incX);
        try {
            xCPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return  ret2 - 1;
    }

    @Override
    protected int isamax(int N, DataBuffer X, int offsetX, int incX){
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected int idamax(int N, INDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);

        int ret2 = JCublas.cublasIdamax(
                N,
                xCPointer.getDevicePointer(),
                incX);
        try {
            xCPointer.close();
            ctx.finishBlasOperation();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return  ret2 - 1;
    }

    @Override
    protected int idamax(int N, DataBuffer X, int offsetX, int incX){
        throw new UnsupportedOperationException("not yet implemented");
    }


    @Override
    protected int icamax(int N, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        int[] result = new int[1];
        Pointer resultPointer = Pointer.to(result);
        ctx.finishBlasOperation();
        try {
            xCPointer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return JCublas2.cublasIcamax(ContextHolder.getInstance().getHandle(),N,xCPointer.getDevicePointer(),incX,resultPointer) - 1;
    }

    @Override
    protected int izamax(int N, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        int[] result = new int[1];
        Pointer resultPointer = Pointer.to(result);
        ctx.finishBlasOperation();
        try {
            xCPointer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return JCublas2.cublasIzamax(ContextHolder.getInstance().getHandle(), N, xCPointer.getDevicePointer(), incX, resultPointer) - 1;
    }

    @Override
    protected void sswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        CublasPointer yCPointer = new CublasPointer(Y,ctx);

        JCublas2.cublasSswap(
                ctx.getHandle(),
                N,
                xCPointer.getDevicePointer(),
                incX,
                yCPointer.getDevicePointer(),
                incY);
        yCPointer.copyToHost();

        try {
            xCPointer.close();
            yCPointer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        ctx.finishBlasOperation();

    }

    @Override
    protected void scopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xCPointer = new CublasPointer(X,ctx);
        CublasPointer yCPointer = new CublasPointer(Y,ctx);

        JCublas2.cublasScopy(
                ctx.getHandle()
                , N, xCPointer.getDevicePointer()
                , incX
                , yCPointer.getDevicePointer()
                , incY);


        yCPointer.copyToHost();
        try {
            xCPointer.close();
            yCPointer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        ctx.finishBlasOperation();
    }

    @Override
    protected void scopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void saxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(X,ctx);
        CublasPointer xBPointer = new CublasPointer(Y,ctx);


        JCublas2.cublasSaxpy(
                ctx.getHandle(),
                N,
                Pointer.to(new float[]{alpha}),
                xAPointer.getDevicePointer(),
                incX,
                xBPointer.getDevicePointer(),
                incY);



        xBPointer.copyToHost();
        try {
            xAPointer.close();
            xBPointer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        ctx.finishBlasOperation();


    }

    @Override
    protected void saxpy( int N, float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void dswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xCPointer = new CublasPointer(X,ctx);
        CublasPointer yCPointer = new CublasPointer(Y,ctx);



        JCublas2.cublasDswap(
                ctx.getHandle(),
                N,
                xCPointer.getDevicePointer(),
                incX,
                yCPointer.getDevicePointer(),
                incY);

        yCPointer.copyToHost();
        try {
            xCPointer.close();
            yCPointer.close();
            ctx.finishBlasOperation();
        }catch (Exception e) {
            e.printStackTrace();
        }


    }

    @Override
    protected void dcopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xCPointer = new CublasPointer(X,ctx);
        CublasPointer yCPointer = new CublasPointer(Y,ctx);

        JCublas2.cublasDcopy(
                ctx.getHandle()
                , N, xCPointer.getDevicePointer()
                , incX
                , yCPointer.getDevicePointer()
                , incY);


        yCPointer.copyToHost();
        try {
            xCPointer.close();
            yCPointer.close();
            ctx.finishBlasOperation();
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void dcopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void daxpy(int N, double alpha, INDArray X, int incX, INDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(X,ctx);
        CublasPointer xBPointer = new CublasPointer(Y,ctx);


        JCublas2.cublasDaxpy(
                ctx.getHandle(),
                N,
                Pointer.to(new double[]{alpha}),
                xAPointer.getDevicePointer(),
                incX,
                xBPointer.getDevicePointer(),
                incY);



        xBPointer.copyToHost();
        ctx.finishBlasOperation();

        try {
            xAPointer.close();
            xBPointer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void daxpy( int N, double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void cswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(X,ctx);
        CublasPointer xBPointer = new CublasPointer(Y,ctx);

        JCublas2.cublasCswap(ContextHolder.getInstance().getHandle(), N, xAPointer.getDevicePointer(), incX, xBPointer.getDevicePointer(), incY);

        xBPointer.copyToHost();
        ctx.finishBlasOperation();
        CublasPointer.free(xAPointer, xBPointer);

    }

    @Override
    protected void ccopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void caxpy(int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer aCPointer = new CublasPointer(X,ctx);
        CublasPointer bCPointer = new CublasPointer(Y,ctx);


        JCublas2.cublasCaxpy(
                ctx.getHandle(),
                N,
                PointerUtil.getPointer(jcuda.cuComplex.cuCmplx(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue())),
                aCPointer.getDevicePointer(),
                incX,
                bCPointer.getDevicePointer(),
                incY
        );

        ctx.finishBlasOperation();
        CublasPointer.free(aCPointer, bCPointer);




    }

    @Override
    protected void zswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(X,ctx);
        CublasPointer xBPointer = new CublasPointer(Y,ctx);

        JCublas2.cublasZswap(ContextHolder.getInstance().getHandle(), N, xAPointer.getDevicePointer(), incX, xBPointer.getDevicePointer(), incY);

        xBPointer.copyToHost();
        CublasPointer.free(xAPointer,xBPointer);
        ctx.finishBlasOperation();

    }

    @Override
    protected void zcopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void zaxpy(int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer aCPointer = new CublasPointer(X,ctx);
        CublasPointer bCPointer = new CublasPointer(Y,ctx);


        JCublas2.cublasZaxpy(
                ctx.getHandle(),
                N,
                PointerUtil.getPointer(jcuda.cuComplex.cuCmplx(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue())),
                aCPointer.getDevicePointer(),
                incX,
                bCPointer.getDevicePointer(),
                incY
        );

        ctx.finishBlasOperation();
        CublasPointer.free(aCPointer,bCPointer);


    }

    @Override
    protected void srotg(float a, float b, float c, float s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srotmg(float d1, float d2, float b1, float b2, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srot(int N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drotg(double a, double b, double c, double s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drotmg(double d1, double d2, double b1, double b2, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drot(int N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sscal(int N, float alpha, INDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        JCublas2.cublasSscal(
                ctx.getHandle(),
                N,
                Pointer.to(new float[]{alpha}),
                xCPointer.getDevicePointer(),
                incX);


        xCPointer.copyToHost();
        CublasPointer.free(xCPointer);


    }

    @Override
    protected void dscal(int N, double alpha, INDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);
        JCublas2.cublasDscal(
                ctx.getHandle(),
                N,
                Pointer.to(new double[]{alpha}),
                xCPointer.getDevicePointer(),
                incX);


        xCPointer.copyToHost();
        CublasPointer.free(xCPointer);

    }

    @Override
    protected void cscal(int N, IComplexFloat alpha, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);

        JCublas2.cublasCscal(
                ctx.getHandle(),
                N,
                PointerUtil.getPointer(jcuda.cuComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent())),
                xCPointer.getDevicePointer(),
                incX
        );

        ctx.finishBlasOperation();
        xCPointer.copyToHost();
        CublasPointer.free(xCPointer);

    }

    @Override
    protected void zscal(int N, IComplexDouble alpha, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer xCPointer = new CublasPointer(X,ctx);

        JCublas2.cublasZscal(
                ctx.getHandle(),
                N,
                PointerUtil.getPointer(jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent())),
                xCPointer.getDevicePointer(),
                incX
        );

        xCPointer.copyToHost();
        ctx.finishBlasOperation();
        CublasPointer.free(xCPointer);


    }

    @Override
    protected void csscal(int N, float alpha, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer p = new CublasPointer(X,ctx);
        JCublas2.cublasSscal(ContextHolder.getInstance().getHandle(), N, Pointer.to(new float[]{alpha}), p.getDevicePointer(), incX);
        p.copyToHost();
    }

    @Override
    protected void zdscal(int N, double alpha, IComplexNDArray X, int incX) {
        CudaContext ctx = CudaContext.getBlasContext();
        CublasPointer p = new CublasPointer(X,ctx);
        JCublas2.cublasZdscal(ContextHolder.getInstance().getHandle(), N, Pointer.to(new double[]{alpha}), p.getDevicePointer(), incX);
        p.copyToHost();
        ctx.finishBlasOperation();
        CublasPointer.free(p);
    }

    @Override
    public boolean supportsDataBufferL1Ops() {
        return false;
    }
}
