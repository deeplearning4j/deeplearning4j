package org.nd4j.linalg.cpu.nativecpu.blas;


import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Dot;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.Nd4jBlas;

import static org.bytedeco.javacpp.openblas.*;


/**
 * @author Adam Gibson
 */
public class CpuLevel1 extends BaseLevel1 {
    private Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();

    @Override
    protected float sdsdot(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        return cblas_sdsdot(N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected double dsdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return cblas_dsdot(N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected float hdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float hdot(int N, DataBuffer X, int offsetX, int incX, DataBuffer Y, int offsetY, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float sdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        if (incX >= 1 && incY >= 1) {
            return cblas_sdot(N, (FloatPointer) X.data().addressPointer(), incX,
                            (FloatPointer) Y.data().addressPointer(), incY);
        } else {
            // non-EWS dot variant
            Dot dot = new Dot(X, Y);
            Nd4j.getExecutioner().exec(dot);
            return dot.getFinalResult().floatValue();
        }
    }

    @Override
    protected float sdot(int N, DataBuffer X, int offsetX, int incX, DataBuffer Y, int offsetY, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected double ddot(int N, INDArray X, int incX, INDArray Y, int incY) {
        if (incX >= 1 && incY >= 1) {
            return cblas_ddot(N, (DoublePointer) X.data().addressPointer(), incX,
                            (DoublePointer) Y.data().addressPointer(), incY);
        } else {
            // non-EWS dot variant
            Dot dot = new Dot(X, Y);
            Nd4j.getExecutioner().exec(dot);
            return dot.getFinalResult().doubleValue();
        }
    }

    @Override
    protected double ddot(int N, DataBuffer X, int offsetX, int incX, DataBuffer Y, int offsetY, int incY) {
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
        return cblas_snrm2(N, (FloatPointer) X.data().addressPointer(), incX);

    }

    @Override
    protected float sasum(int N, INDArray X, int incX) {
        return cblas_sasum(N, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected float sasum(int N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incX) {
        return cblas_dnrm2(N, (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected double dasum(int N, INDArray X, int incX) {
        return cblas_dasum(N, (DoublePointer) X.data().addressPointer(), incX);
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
        return (int) cblas_isamax(N, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected int isamax(int N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected int idamax(int N, INDArray X, int incX) {
        return (int) cblas_idamax(N, (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected int idamax(int N, DataBuffer X, int offsetX, int incX) {
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
        cblas_sswap(N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void scopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        cblas_scopy(N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void scopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void haxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void saxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        cblas_saxpy(N, alpha, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    public void haxpy(int n, float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void saxpy(int n, float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }


    @Override
    protected void dswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        cblas_dswap(N, (DoublePointer) X.data().addressPointer(), incX, (DoublePointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected void dcopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        cblas_dcopy(N, (DoublePointer) X.data().addressPointer(), incX, (DoublePointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected void dcopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void daxpy(int N, double alpha, INDArray X, int incX, INDArray Y, int incY) {
        cblas_daxpy(N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Y.data().addressPointer(), incY);

    }

    @Override
    public void daxpy(int n, double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
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
        cblas_srotmg(new FloatPointer(d1), new FloatPointer(d2), new FloatPointer(b1), b2,
                        (FloatPointer) P.data().addressPointer());
    }

    @Override
    protected void srot(int N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {
        cblas_srot(N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(), incY, c,
                        s);
    }

    @Override
    protected void srotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        cblas_srotm(N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(), incY,
                        (FloatPointer) P.data().addressPointer());

    }

    @Override
    protected void drotg(double a, double b, double c, double s) {
        cblas_drotg(new DoublePointer(a), new DoublePointer(b), new DoublePointer(c), new DoublePointer(s));
    }

    @Override
    protected void drotmg(double d1, double d2, double b1, double b2, INDArray P) {
        cblas_drotmg(new DoublePointer(d1), new DoublePointer(d2), new DoublePointer(b1), b2,
                        (DoublePointer) P.data().addressPointer());
    }

    @Override
    protected void drot(int N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {
        cblas_drot(N, (DoublePointer) X.data().addressPointer(), incX, (DoublePointer) Y.data().addressPointer(), incY,
                        c, s);
    }


    @Override
    protected void drotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        cblas_drotm(N, (DoublePointer) X.data().addressPointer(), incX, (DoublePointer) Y.data().addressPointer(), incY,
                        (DoublePointer) P.data().addressPointer());
    }

    @Override
    protected void sscal(int N, float alpha, INDArray X, int incX) {
        cblas_sscal(N, alpha, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void dscal(int N, double alpha, INDArray X, int incX) {
        cblas_dscal(N, alpha, (DoublePointer) X.data().addressPointer(), incX);
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
