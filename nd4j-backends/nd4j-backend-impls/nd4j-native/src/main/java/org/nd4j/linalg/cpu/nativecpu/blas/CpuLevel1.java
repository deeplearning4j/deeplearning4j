package org.nd4j.linalg.cpu.nativecpu.blas;


import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.Nd4jBlas;



/**
 * @author Adam Gibson
 */
public class CpuLevel1 extends BaseLevel1 {
    private Nd4jBlas nd4jBlas = NativeOpsHolder.getInstance().getDeviceNativeBlas();
    private static PointerPointer DUMMY = new PointerPointer(new Pointer[] {null});
    @Override
    protected float sdsdot(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        return nd4jBlas.sdsdot(DUMMY,N,alpha,(FloatPointer)X.data().addressPointer(),incX,(FloatPointer)Y.data().addressPointer(),incY);
    }

    @Override
    protected double dsdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return nd4jBlas.dsdot(DUMMY,N,(FloatPointer)X.data().addressPointer(),incX,(FloatPointer)Y.data().addressPointer(),incY);
    }

    @Override
    protected float hdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float hdot( int N, DataBuffer X, int offsetX, int incX, DataBuffer Y,  int offsetY, int incY){
        throw new UnsupportedOperationException();
    }

    @Override
    protected float sdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return nd4jBlas.sdot(DUMMY,N,(FloatPointer)X.data().addressPointer(),incX,(FloatPointer)Y.data().addressPointer(),incY);
    }

    @Override
    protected float sdot( int N, DataBuffer X, int offsetX, int incX, DataBuffer Y,  int offsetY, int incY){
        throw new UnsupportedOperationException();
    }

    @Override
    protected double ddot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return nd4jBlas.ddot(DUMMY,N,(DoublePointer)X.data().addressPointer(),incX,(DoublePointer)Y.data().addressPointer(),incY);
    }

    @Override
    protected double ddot( int N, DataBuffer X, int offsetX, int incX, DataBuffer Y,  int offsetY, int incY){
        throw new UnsupportedOperationException();
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
        return nd4jBlas.snrm2(DUMMY,N,(FloatPointer)X.data().addressPointer(),incX);

    }

    @Override
    protected float sasum(int N, INDArray X, int incX) {
        return nd4jBlas.sasum(DUMMY,N,(FloatPointer)X.data().addressPointer(),incX);
    }

    @Override
    protected float sasum(int N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incX) {
        return nd4jBlas.dnrm2(DUMMY,N,(DoublePointer)X.data().addressPointer(),incX);
    }

    @Override
    protected double dasum(int N, INDArray X, int incX) {
        return nd4jBlas.dasum(DUMMY,N,(DoublePointer)X.data().addressPointer(),incX);
    }

    @Override
    protected double dasum(int N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
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
        return nd4jBlas.isamax(DUMMY,N,(FloatPointer)X.data().addressPointer(),incX);
    }

    @Override
    protected int isamax(int N, DataBuffer X, int offsetX, int incX){
        throw new UnsupportedOperationException();
    }

    @Override
    protected int idamax(int N, INDArray X, int incX) {
        return nd4jBlas.idamax(DUMMY,N,(DoublePointer)X.data().addressPointer(),incX);
    }

    @Override
    protected int idamax(int N, DataBuffer X, int offsetX, int incX){
        throw new UnsupportedOperationException();
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
        nd4jBlas.sswap(DUMMY,N,(FloatPointer)X.data().addressPointer(),incX,(FloatPointer)Y.data().addressPointer(),incY);
    }

    @Override
    protected void scopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        nd4jBlas.scopy(DUMMY,N,(FloatPointer)X.data().addressPointer(),incX,(FloatPointer)Y.data().addressPointer(),incY);
    }

    @Override
    protected void scopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        throw new UnsupportedOperationException();
    }

    @Override
    protected void haxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void saxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        nd4jBlas.saxpy(DUMMY,N,alpha,(FloatPointer)X.data().addressPointer(),incX,(FloatPointer)Y.data().addressPointer(),incY);
    }

    @Override
    public void haxpy(int n,float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        throw new UnsupportedOperationException();
    }

    @Override
    public void saxpy(int n,float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        throw new UnsupportedOperationException();
    }


    @Override
    protected void dswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        nd4jBlas.dswap(DUMMY,N,(DoublePointer)X.data().addressPointer(),incX,(DoublePointer)Y.data().addressPointer(),incY);
    }

    @Override
    protected void dcopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        nd4jBlas.dcopy(DUMMY,N,(DoublePointer)X.data().addressPointer(),incX,(DoublePointer)Y.data().addressPointer(),incY);
    }

    @Override
    protected void dcopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void daxpy(int N, double alpha, INDArray X, int incX, INDArray Y, int incY) {
        nd4jBlas.daxpy(DUMMY,N,alpha,(DoublePointer)X.data().addressPointer(),incX,(DoublePointer)Y.data().addressPointer(),incY);

    }

    @Override
    public void daxpy(int n,double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY){
        throw new UnsupportedOperationException();
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
        DataBuffer addr = Nd4j.createBuffer(new float[]{d1,d2,b1,b2});
        nd4jBlas.srotmg(DUMMY,(FloatPointer)addr.addressPointer(),(FloatPointer)P.data().addressPointer());
    }

    @Override
    protected void srot(int N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {
        nd4jBlas.srot(DUMMY,N,(FloatPointer)X.data().addressPointer(),incX,(FloatPointer)Y.data().addressPointer(),incY,c,s);
    }

    @Override
    protected void srotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        nd4jBlas.srotm(DUMMY,N,(FloatPointer)X.data().addressPointer(),incX,(FloatPointer)Y.data().addressPointer(),incY,(FloatPointer)P.data().addressPointer());

    }

    @Override
    protected void drotg(double a, double b, double c, double s) {
        DataBuffer buff = Nd4j.createBuffer(new double[]{a, b, c, s});
        nd4jBlas.drotg(DUMMY,(DoublePointer)buff.addressPointer());
    }

    @Override
    protected void drotmg(double d1, double d2, double b1, double b2, INDArray P) {
        DataBuffer buff = Nd4j.createBuffer(new double[]{d1, d2, b1, b2});
        nd4jBlas.drotmg(DUMMY,(DoublePointer)buff.addressPointer(),(DoublePointer)P.data().addressPointer());
    }

    @Override
    protected void drot(int N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {
        nd4jBlas.drot(DUMMY,N,(DoublePointer)X.data().addressPointer(),incX,(DoublePointer)Y.data().addressPointer(),incY,c,s);
    }


    @Override
    protected void drotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        nd4jBlas.drotm(DUMMY,N,(DoublePointer)X.data().addressPointer(),incX,(DoublePointer)Y.data().addressPointer(),incY,(DoublePointer)P.data().addressPointer());
    }

    @Override
    protected void sscal(int N, float alpha, INDArray X, int incX) {
        nd4jBlas.sscal(DUMMY,N,alpha,(FloatPointer)X.data().addressPointer(),incX);
    }

    @Override
    protected void dscal(int N, double alpha, INDArray X, int incX) {
        nd4jBlas.dscal(DUMMY,N,alpha,(DoublePointer)X.data().addressPointer(),incX);
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

    @Override
    protected float hasum(int N, INDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float hasum(int N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
    }
}
