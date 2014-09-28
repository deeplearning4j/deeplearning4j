package org.nd4j.linalg.netlib;

import com.github.fommil.netlib.BLAS;
import org.jblas.NativeBlas;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.netlib.complex.ComplexDouble;
import org.nd4j.linalg.netlib.complex.ComplexFloat;

/**
 *
 * @author Adam Gibson
 *
 */
public class SimpleNetlibBlas {




    /**
     * General matrix vector multiplication
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, float alpha, float beta) {
        BLAS.getInstance().sgemv(
                "N",
                A.rows(),
                A.columns(),
                alpha,
                A.data(),
                A.offset(),
                A.rows(),
                B.data(),
                B.offset(),
                C.stride()[0],
                beta,
                C.data(),
                C.offset(),
                C.stride()[0]);



        return C;
    }


    /**
     * General matrix multiply
     * @param A
     * @param B
     * @param a
     * @param C
     * @param b
     * @return
     */
    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexNumber a,IComplexNDArray C
            , IComplexNumber b) {

        NativeBlas.cgemm(
                'N',
                'N',
                C.rows(),
                C.columns(),
                A.columns(),
                new ComplexFloat(a.realComponent().floatValue(), a.imaginaryComponent().floatValue()),
                A.data(),
                A.offset() / 2,
                A.rows(),
                B.data(),
                B.offset() / 2,
                B.rows(),
                new ComplexFloat(b.realComponent().floatValue(), b.imaginaryComponent().floatValue())
                ,
                C.data(),
                C.offset() / 2,
                C.rows());
        return C;

    }


    /**
     * General matrix multiply
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemm(INDArray A, INDArray B, INDArray C,
                                float alpha, float beta) {



        BLAS.getInstance().sgemm(
                "n",
                "n",
                C.rows(),
                C.columns(),
                A.columns(),
                alpha,
                A.data(),
                A.rows(),
                B.columns()
                , B.data(),
                B.rows(),
                C.rows(),
                beta,
                C.data(),
                C.rows(),
                C.columns());

        return C;

    }


    /**
     * Calculate the 2 norm of the ndarray
     * @param A
     * @return
     */
    public static float nrm2(IComplexNDArray A) {
        float s = BLAS.getInstance().snrm2(A.length(), A.data(), 2, 1);
        return s;
    }

    /**
     * Copy x to y
     * @param x the origin
     * @param y the destination
     */
    public static void copy(IComplexNDArray x, IComplexNDArray y) {


        BLAS.getInstance().scopy(
                x.length(),
                x.data(),
                x.stride()[0],
                y.data(),
                y.stride()[0]);
    }


    /**
     * Return the index of the max in the given ndarray
     * @param x the ndarray to ge tthe max for
     * @return
     */
    public static int iamax(IComplexNDArray x) {
        return NativeBlas.icamax(x.length(), x.data(), x.offset(), 1) - 1;
    }

    /**
     *
     * @param x
     * @return
     */
    public static float asum(IComplexNDArray x) {

        return NativeBlas.scasum(x.length(), x.data(), x.offset(), x.stride()[0]);
    }


    /**
     * Swap the elements in each ndarray
     * @param x
     * @param y
     */
    public static void swap(INDArray x, INDArray y) {

        BLAS.getInstance().sswap(
                x.length(),
                x.data(),
                x.offset(),
                x.stride()[0],
                y.data(),
                y.offset(),
                y.stride()[0]);

    }

    /**
     *
     * @param x
     * @return
     */
    public static float asum(INDArray x) {
        float sum = BLAS.getInstance().sasum(x.length(), x.data(), x.offset(),x.stride()[0]);
        return sum;
    }

    /**
     * Returns the norm2 of the given ndarray
     * @param x
     * @return
     */
    public static float nrm2(INDArray x) {
        float normal2 = BLAS.getInstance().snrm2(x.length(), x.data(), x.offset(),x.stride()[0]);
        return normal2;
    }

    /**
     * Returns the index of the max element
     * in the given ndarray
     * @param x
     * @return
     */
    public static int iamax(INDArray x) {
        int max =  BLAS.getInstance().isamax(
                x.length(),
                x.data(),
                x.offset(),
                x.stride()[0]);
        return max;

    }

    /**
     * And and scale by the given scalar da
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(float da, INDArray A, INDArray B) {

        if(A.ordering() == NDArrayFactory.C) {

            BLAS.getInstance().saxpy(
                    A.length(),
                    da,
                    A.data(),
                    A.offset(),
                    A.stride()[0],
                    B.data(),
                    B.offset(),
                    B.stride()[0]);
        }
        else {
            BLAS.getInstance().saxpy(
                    A.length(),
                    da,
                    A.data(),
                    A.offset(),
                    A.stride()[0],
                    B.data(),
                    B.offset(),
                    B.stride()[0]);

        }



    }


    /**
     *
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexNumber da, IComplexNDArray A, IComplexNDArray B) {
        NativeBlas.caxpy(A.length(), new org.jblas.ComplexFloat(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()), A.data(), A.offset(), 1, B.data(), B.offset(), 1);


    }

    /**
     * Multiply the given ndarray
     *  by alpha
     * @param alpha
     * @param x
     * @return
     */
    public static INDArray scal(float alpha, INDArray x) {
        BLAS.getInstance().sscal(
                x.length(),
                alpha,
                x.data(),
                x.offset(),
                x.stride()[0]);

        return x;

    }

    /**
     * Copy x to y
     * @param x
     * @param y
     */
    public static void copy(INDArray x, INDArray y) {
        BLAS.getInstance().scopy(x.length(),
                x.data(),
                x.offset(),
                x.stride()[0],
                y.data(),
                y.offset(),
                y.stride()[0]);


    }

    /**
     * Dot product between 2 ndarrays
     * @param x
     * @param y
     * @return
     */
    public static float dot(INDArray x, INDArray y) {
        float ret =  BLAS.getInstance().sdot(
                x.length(),
                x.data(),
                x.offset(),
                x.stride()[0],
                y.data(),
                y.offset(),
                y.stride()[0]);

        return ret;
    }


    /**
     * Dot product between two complex ndarrays
     * @param x
     * @param y
     * @return
     */
    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {
        ComplexFloat f = (ComplexFloat) NativeBlas.cdotc(x.length(), x.data(), x.offset(), 1, y.data(), y.offset(), 1);
        return new ComplexDouble(f.realComponent().doubleValue(),f.imaginaryComponent().doubleValue());

    }



    public static INDArray ger(INDArray A, INDArray B, INDArray C, float alpha) {


        // = alpha * A * transpose(B) + C
        BLAS.getInstance().sger(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                A.data(),        // d_A or x
                A.rows(),   // incx
                B.data(),        // dB or y
                B.rows(),   // incy
                C.data(),        // dC or A
                C.rows()    // lda
        );


        return C;
    }





    /**
     * Complex dot product
     * @param x
     * @param y
     * @return
     */
    public static IComplexFloat dotu(IComplexNDArray x, IComplexNDArray y) {
        return new ComplexFloat(NativeBlas.cdotu(x.length(), x.data(), x.offset(), 1, y.data(), y.offset(), 1));
    }

    /**
     *
     * @param alpha
     * @param x
     * @param y
     * @param a
     * @return
     */
    public static IComplexNDArray geru(IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        NativeBlas.cgeru(a.rows(), a.columns(),
                new ComplexFloat(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue()),
                x.data(), x.offset(), 1, y.data(), y.offset(), 1, a.data(),
                a.offset(), a.rows());
        return a;
    }

    /**
     *
     * @param x
     * @param y
     * @param a
     * @param alpha
     * @return
     */
    public static IComplexNDArray gerc(IComplexNDArray x, IComplexNDArray y, IComplexNDArray a,
                                       IComplexDouble alpha) {
        NativeBlas.cgerc(a.rows(), a.columns(), (ComplexFloat) alpha, x.data(), x.offset(), 1, y.data(), y.offset(), 1, a.data(),
                a.offset(), a.rows());
        return a;
    }


    /**
     * Simpler version of saxpy
     * taking in to account the parameters of the ndarray
     * @param alpha the alpha to scale by
     * @param x the x
     * @param y the y
     */
    public static void saxpy(float alpha, INDArray x, INDArray y) {
        axpy(alpha,x,y);
    }

    /**
     * Scale a complex ndarray
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray sscal(IComplexFloat alpha, IComplexNDArray x) {
        NativeBlas.cscal(x.length(),(org.jblas.ComplexFloat) alpha,x.data(),x.offset(),x.stride()[0]);
        return x;
    }
}
