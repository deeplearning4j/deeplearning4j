package org.nd4j.linalg.cpu.nativecpu.blas;


import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.CBLAS;


import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;



/**
 * @author Adam Gibson
 */
public class CpuLevel1 extends BaseLevel1 {
    @Override
    protected float sdsdot(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        return CBLAS.sdsdot(N,alpha,X.data().asNioFloat(),incX,Y.data().asNioFloat(),incY);
    }

    @Override
    protected double dsdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return CBLAS.dsdot(N,X.data().asNioDouble(),incX,Y.data().asNioDouble(),incY);
    }

    @Override
    protected float sdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return CBLAS.sdot(N,X.data().asNioFloat(),incX,Y.data().asNioFloat(),incY);
    }

    @Override
    protected float sdot( int N, DataBuffer X, int offsetX, int incX, DataBuffer Y,  int offsetY, int incY){
        return CBLAS.sdot(N, X.asNioFloat(), incX, Y.asNioFloat(), incY);
    }

    @Override
    protected double ddot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return CBLAS.ddot(N,X.data().asNioDouble(),incX,Y.data().asNioDouble(),incY);
    }

    @Override
    protected double ddot( int N, DataBuffer X, int offsetX, int incX, DataBuffer Y,  int offsetY, int incY){
        return CBLAS.ddot(N, X.asNioDouble(), incX, Y.asNioDouble(), incY);
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
        return CBLAS.snrm2(N,X.data().asNioFloat(),incX);

    }

    @Override
    protected float sasum(int N, INDArray X, int incX) {
        return CBLAS.sasum(N, X.data().asNioFloat(), incX);
    }

    @Override
    protected float sasum(int N, DataBuffer X, int offsetX, int incX) {
        return CBLAS.sasum(N,X.asNioFloat(),incX);
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incX) {
        return CBLAS.dnrm2(N,X.data().asNioDouble(),incX);
    }

    @Override
    protected double dasum(int N, INDArray X, int incX) {
        return CBLAS.dasum(N,X.data().asNioDouble(),incX);
    }

    @Override
    protected double dasum(int N, DataBuffer X, int offsetX, int incX) {
        return CBLAS.dasum(N,X.asNioDouble(),incX);
    }

    @Override
    protected float scnrm2(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float scasum(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected double dznrm2(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected double dzasum(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected int isamax(int N, INDArray X, int incX) {
        return CBLAS.isamax(N,X.data().asNioFloat(),incX);
    }

    @Override
    protected int isamax(int N, DataBuffer X, int offsetX, int incX){
        return CBLAS.isamax(N,X.asNioFloat(),incX);
    }

    @Override
    protected int idamax(int N, INDArray X, int incX) {
        return CBLAS.idamax(N, X.data().asNioDouble(), incX);
    }

    @Override
    protected int idamax(int N, DataBuffer X, int offsetX, int incX){
        return CBLAS.idamax(N, X.asNioDouble(), incX);
    }

    @Override
    protected int icamax(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected int izamax(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void sswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        CBLAS.sswap(N,X.data().asNioFloat(),incX,Y.data().asNioFloat(),incY);
    }

    @Override
    protected void scopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        CBLAS.scopy(N, X.data().asNioFloat(), incX, Y.data().asNioFloat(), incY);

    }

    @Override
    protected void scopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        CBLAS.scopy(n, x.asNioFloat(), incrX, y.asNioFloat(), incrY);
    }

    @Override
    protected void saxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        CBLAS.saxpy(N,alpha,X.data().asNioFloat(),incX,Y.data().asNioFloat(),incY);
    }

    @Override
    public void saxpy(int n,float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        CBLAS.saxpy(n, alpha, x.asNioFloat(), incrX, y.asNioFloat(), incrY);
    }


    @Override
    protected void dswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        CBLAS.dswap(N,X.data().asNioDouble(),incX,Y.data().asNioDouble(),incY);
    }

    @Override
    protected void dcopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        CBLAS.dcopy(N,X.data().asNioDouble(),incX,Y.data().asNioDouble(),incY);
    }

    @Override
    protected void dcopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        CBLAS.dcopy(n,x.asNioDouble(),incrX,y.asNioDouble(),incrY);
    }

    @Override
    protected void daxpy(int N, double alpha, INDArray X, int incX, INDArray Y, int incY) {
        CBLAS.daxpy(N,alpha,X.data().asNioDouble(),incX,Y.data().asNioDouble(),incY);
    }

    @Override
    public void daxpy(int n,double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY){
        CBLAS.daxpy(n,alpha,x.asNioDouble(),incrX,y.asNioDouble(),incrY);
    }

    @Override
    protected void cswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void ccopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void caxpy(int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zcopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zaxpy(int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srotg(float a, float b, float c, float s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srotmg(float d1, float d2, float b1, float b2, INDArray P) {
        FloatBuffer wrap = FloatBuffer.wrap(new float[]{d1,d2,b1,b2});
        CBLAS.srotmg(wrap,P.data().asNioFloat());
    }

    @Override
    protected void srot(int N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {
        CBLAS.srot(N,X.data().asNioFloat(),incX,Y.data().asNioFloat(),incY,c,s);
    }

    @Override
    protected void srotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        CBLAS.srotm(N,X.data().asNioFloat(),incX,Y.data().asNioFloat(),incY,P.data().asNioFloat());

    }

    @Override
    protected void drotg(double a, double b, double c, double s) {
        DoubleBuffer buff = DoubleBuffer.wrap(new double[]{a,b,c,s});
        CBLAS.drotg(buff);
    }

    @Override
    protected void drotmg(double d1, double d2, double b1, double b2, INDArray P) {
        DoubleBuffer buff = DoubleBuffer.wrap(new double[]{d1,d2,b1,b2});
        CBLAS.drotmg(buff,P.data().asNioDouble());
    }

    @Override
    protected void drot(int N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {
        CBLAS.drot(N,X.data().asNioDouble(),incX,Y.data().asNioDouble(),incY,c,s);
    }


    @Override
    protected void drotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        CBLAS.drotm(N,X.data().asNioDouble(),incX,Y.data().asNioDouble(),incY,P.data().asNioDouble());
    }

    @Override
    protected void sscal(int N, float alpha, INDArray X, int incX) {
        CBLAS.sscal(N,alpha,X.data().asNioFloat(),incX);
    }

    @Override
    protected void dscal(int N, double alpha, INDArray X, int incX) {
        CBLAS.dscal(N,alpha,X.data().asNioDouble(),incX);
    }

    @Override
    protected void cscal(int N, IComplexFloat alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void zscal(int N, IComplexDouble alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void csscal(int N, float alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zdscal(int N, double alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }
}
