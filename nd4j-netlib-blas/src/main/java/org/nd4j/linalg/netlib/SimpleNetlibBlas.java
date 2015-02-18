/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.netlib;

import com.github.fommil.netlib.BLAS;
import com.github.fommil.netlib.LAPACK;
import org.jblas.NativeBlas;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.netlib.complex.ComplexDouble;
import org.nd4j.linalg.netlib.complex.ComplexFloat;
import org.netlib.util.intW;

/**
 * @author Adam Gibson
 */
public class SimpleNetlibBlas {


    /**
     * General matrix vector multiplication
     *
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, double alpha, double beta) {
        BLAS.getInstance().dgemv(
                "N",
                A.rows(),
                A.columns(),
                alpha,
                A.data().asDouble(),
                A.offset(),
                A.rows(),
                B.data().asDouble(),
                B.offset(),
                C.majorStride(),
                beta,
                C.data().asDouble(),
                C.offset(),
                C.majorStride());


        return C;
    }


    /**
     * General matrix vector multiplication
     *
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, float alpha, float beta) {
        DataTypeValidation.assertFloat(A, B, C);
        BLAS.getInstance().sgemv(
                "N",
                A.rows(),
                A.columns(),
                alpha,
                A.data().asFloat(),
                A.offset(),
                A.rows(),
                B.data().asFloat(),
                B.offset(),
                C.majorStride(),
                beta,
                C.data().asFloat(),
                C.offset(),
                C.majorStride());


        return C;
    }


    /**
     * General matrix multiply
     *
     * @param A
     * @param B
     * @param a
     * @param C
     * @param b
     * @return
     */
    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexNumber a, IComplexNDArray C
            , IComplexNumber b) {
        DataTypeValidation.assertSameDataType(A, B, C);
        if (A.data().dataType() == DataBuffer.FLOAT)
            NativeBlas.cgemm(
                    'N',
                    'N',
                    C.rows(),
                    C.columns(),
                    A.columns(),
                    new ComplexFloat(a.realComponent().floatValue(), a.imaginaryComponent().floatValue()),
                    A.data().asFloat(),
                    A.offset() / 2,
                    A.rows(),
                    B.data().asFloat(),
                    B.offset() / 2,
                    B.rows(),
                    new ComplexFloat(b.realComponent().floatValue(), b.imaginaryComponent().floatValue())
                    ,
                    C.data().asFloat(),
                    C.offset() / 2,
                    C.rows());
        else if (A.data().dataType() == DataBuffer.DOUBLE) {
            NativeBlas.zgemm(
                    'N',
                    'N',
                    C.rows(),
                    C.columns(),
                    A.columns(),
                    new ComplexDouble(a.realComponent().doubleValue(), a.imaginaryComponent().doubleValue()),
                    A.data().asDouble(),
                    A.offset() / 2,
                    A.rows(),
                    B.data().asDouble(),
                    B.offset() / 2,
                    B.rows(),
                    new ComplexDouble(b.realComponent().doubleValue(), b.imaginaryComponent().doubleValue())
                    ,
                    C.data().asDouble(),
                    C.offset() / 2,
                    C.rows());
        }
        return C;

    }

    /**
     * General matrix multiply
     *
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemm(INDArray A, INDArray B, INDArray C,
                                double alpha, double beta) {


        DataTypeValidation.assertDouble(A, B, C);
        BLAS.getInstance().dgemm(
                "N",
                "N",
                C.rows(),
                C.columns(),
                A.columns(),
                alpha,
                A.data().asDouble(),
                A.offset(),
                A.rows()
                , B.data().asDouble(),
                B.offset(),
                B.rows(),
                beta,
                C.data().asDouble(),
                C.offset(),
                C.rows());

        return C;

    }

    /**
     * General matrix multiply
     *
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemm(INDArray A, INDArray B, INDArray C,
                                float alpha, float beta) {

        DataTypeValidation.assertFloat(A, B, C);

        BLAS.getInstance().sgemm(
                "N",
                "N",
                C.rows(),
                C.columns(),
                A.columns(),
                alpha,
                A.data().asFloat(),
                A.offset(),
                A.rows()
                , B.data().asFloat(),
                B.offset(),
                B.rows(),
                beta,
                C.data().asFloat(),
                C.offset(),
                C.rows());

        return C;

    }

    /**
     * Calculate eigen values
     *
     * @param jobz
     * @param range
     * @param uplo
     * @param a
     * @param vl
     * @param vu
     * @param il
     * @param iu
     * @param abstol
     * @param w
     * @param z
     * @param isuppz
     * @return
     */
    public static int syevr(char jobz, char range, char uplo, INDArray a,
                            double vl, int vu, int il, int iu, double abstol,
                            INDArray w, INDArray z, int[] isuppz) {

        DataTypeValidation.assertDouble(a, w, z);
        int n = a.rows();
        org.netlib.util.intW m = new intW(0);
        org.netlib.util.intW info = new org.netlib.util.intW(0);

        double[] work = new double[1];
        int lwork = 0;
        int[] iwork = new int[1];
        int liwork = 0;


        LAPACK.getInstance().dsyevr(
                String.valueOf(jobz),
                String.valueOf(range),
                String.valueOf(uplo),
                n,
                a.data().asDouble(),
                a.offset(),
                a.rows(),
                vl,
                vu,
                il,
                iu,
                abstol,
                m,
                w.data().asDouble(),
                w.offset(),
                z.data().asDouble(),
                z.offset(),
                z.rows(),
                isuppz,
                0,//suppZIdx
                work,
                0,//workOffset
                lwork,
                iwork,
                0,//iworkoffset
                liwork,
                info
        );


        return info.val;

    }

    /**
     * Calculate eigen values
     *
     * @param jobz
     * @param range
     * @param uplo
     * @param a
     * @param vl
     * @param vu
     * @param il
     * @param iu
     * @param abstol
     * @param w
     * @param z
     * @param isuppz
     * @return
     */
    public static int syevr(char jobz, char range, char uplo, INDArray a,
                            float vl, int vu, int il, int iu, float abstol,
                            INDArray w, INDArray z, int[] isuppz) {
        DataTypeValidation.assertFloat(a, w, z);
        int n = a.rows();
        org.netlib.util.intW m = new intW(0);
        org.netlib.util.intW info = new org.netlib.util.intW(0);

        float[] work = new float[1];
        int lwork = 0;
        int[] iwork = new int[1];
        int liwork = 0;


        LAPACK.getInstance().ssyevr(
                String.valueOf(jobz),
                String.valueOf(range),
                String.valueOf(uplo),
                n,
                a.data().asFloat(),
                a.offset(),
                a.rows(),
                vl,
                vu,
                il,
                iu,
                abstol,
                m,
                w.data().asFloat(),
                w.offset(),
                z.data().asFloat(),
                z.offset(),
                z.rows(),
                isuppz,
                0,//suppZIdx
                work,
                0,//workOffset
                lwork,
                iwork,
                0,//iworkoffset
                liwork,
                info
        );


        return info.val;

    }

    /**
     * Calculate the 2 norm of the ndarray
     *
     * @param A
     * @return
     */
    public static double nrm2(IComplexNDArray A) {
        if (A.data().dataType() == DataBuffer.FLOAT) {
            float s = BLAS.getInstance().snrm2(A.length(), A.data().asFloat(), 2, 1);
            return s;
        } else if (A.data().dataType() == DataBuffer.DOUBLE) {
            double s = BLAS.getInstance().dnrm2(A.length(), A.data().asDouble(), 2, 1);
            return s;
        }

        throw new IllegalArgumentException("Illegal data type");


    }

    /**
     * Copy x to y
     *
     * @param x the origin
     * @param y the destination
     */
    public static void copy(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT)

            BLAS.getInstance().scopy(
                    x.length(),
                    x.data().asFloat(),
                    x.majorStride(),
                    y.data().asFloat(),
                    y.majorStride());
        else
            BLAS.getInstance().dcopy(
                    x.length(),
                    x.data().asDouble(),
                    x.majorStride(),
                    y.data().asDouble(),
                    y.majorStride());
    }


    /**
     * Return the index of the max in the given ndarray
     *
     * @param x the ndarray to ge tthe max for
     * @return
     */
    public static int iamax(IComplexNDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT)
            return NativeBlas.icamax(x.length(), x.data().asFloat(), x.offset(), 1) - 1;
        else
            return NativeBlas.izamax(x.length(), x.data().asDouble(), x.offset(), 1) - 1;

    }

    /**
     * @param x
     * @return
     */
    public static double asum(IComplexNDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT)
            return NativeBlas.scasum(x.length(), x.data().asFloat(), x.offset(), x.majorStride());
        else
            return NativeBlas.dzasum(x.length(), x.data().asDouble(), x.offset(), x.majorStride());

    }


    /**
     * Swap the elements in each ndarray
     *
     * @param x
     * @param y
     */
    public static void swap(INDArray x, INDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT)
            BLAS.getInstance().sswap(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.majorStride());
        else
            BLAS.getInstance().dswap(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.majorStride());

    }

    /**
     * @param x
     * @return
     */
    public static double asum(INDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT) {
            float sum = BLAS.getInstance().sasum(x.length(), x.data().asFloat(), x.offset(), x.majorStride());
            return sum;
        } else {
            double sum = BLAS.getInstance().dasum(x.length(), x.data().asDouble(), x.offset(), x.majorStride());
            return sum;
        }
    }

    /**
     * Returns the norm2 of the given ndarray
     *
     * @param x
     * @return
     */
    public static double nrm2(INDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT) {
            float normal2 = BLAS.getInstance().snrm2(x.length(), x.data().asFloat(), x.offset(), x.majorStride());
            return normal2;
        } else {
            double normal2 = BLAS.getInstance().dnrm2(x.length(), x.data().asDouble(), x.offset(), x.majorStride());
            return normal2;
        }

    }

    /**
     * Returns the index of the max element
     * in the given ndarray
     *
     * @param x
     * @return
     */
    public static int iamax(INDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT) {
            int max = BLAS.getInstance().isamax(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.majorStride());
            return max;
        } else {
            int max = BLAS.getInstance().idamax(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.majorStride());
            return max;
        }

    }


    /**
     * Add and scale by the given scalar da
     *
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(double da, INDArray A, INDArray B) {
        DataTypeValidation.assertDouble(A, B);
        if (A.ordering() == NDArrayFactory.C) {

            BLAS.getInstance().daxpy(
                    A.length(),
                    da,
                    A.data().asDouble(),
                    A.offset(),
                    A.majorStride(),
                    B.data().asDouble(),
                    B.offset(),
                    B.majorStride());


        } else {

            BLAS.getInstance().daxpy(
                    A.length(),
                    da,
                    A.data().asDouble(),
                    A.offset(),
                    A.majorStride(),
                    B.data().asDouble(),
                    B.offset(),
                    B.majorStride());

        }


    }

    /**
     * Add and scale by the given scalar da
     *
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(float da, INDArray A, INDArray B) {
        DataTypeValidation.assertFloat(A, B);
        if (A.ordering() == NDArrayFactory.C) {
            BLAS.getInstance().saxpy(
                    A.length(),
                    da,
                    A.data().asFloat(),
                    A.offset(),
                    A.majorStride(),
                    B.data().asFloat(),
                    B.offset(),
                    B.majorStride());


        } else {
            BLAS.getInstance().saxpy(
                    A.length(),
                    da,
                    A.data().asFloat(),
                    A.offset(),
                    A.majorStride(),
                    B.data().asFloat(),
                    B.offset(),
                    B.majorStride());


        }


    }


    /**
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexNumber da, IComplexNDArray A, IComplexNDArray B) {
        DataTypeValidation.assertSameDataType(A, B);
        if (A.data().dataType() == DataBuffer.FLOAT)
            NativeBlas.caxpy(
                    A.length(),
                    new org.jblas.ComplexFloat(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                    A.data().asFloat(),
                    A.offset(),
                    A.majorStride(),
                    B.data().asFloat(),
                    B.offset(),
                    A.majorStride());
        else
            NativeBlas.zaxpy(
                    A.length(),
                    new org.jblas.ComplexDouble(da.realComponent().doubleValue(), da.imaginaryComponent().doubleValue()),
                    A.data().asDouble(),
                    A.offset(),
                    A.majorStride(),
                    B.data().asDouble(),
                    B.offset(),
                    B.majorStride());


    }


    /**
     * Multiply the given ndarray
     * by alpha
     *
     * @param alpha
     * @param x
     * @return
     */
    public static INDArray scal(double alpha, INDArray x) {
        DataTypeValidation.assertDouble(x);
        BLAS.getInstance().dscal(
                x.length(),
                alpha,
                x.data().asDouble(),
                x.offset(),
                x.majorStride());

        return x;

    }

    /**
     * Multiply the given ndarray
     * by alpha
     *
     * @param alpha
     * @param x
     * @return
     */
    public static INDArray scal(float alpha, INDArray x) {
        DataTypeValidation.assertFloat(x);

        BLAS.getInstance().sscal(
                x.length(),
                alpha,
                x.data().asFloat(),
                x.offset(),
                x.majorStride());

        return x;

    }

    /**
     * Copy x to y
     *
     * @param x
     * @param y
     */
    public static void copy(INDArray x, INDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT)
            BLAS.getInstance().scopy(x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.majorStride());
        else
            BLAS.getInstance().dcopy(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.majorStride());


    }

    /**
     * Dot product between 2 ndarrays
     *
     * @param x
     * @param y
     * @return
     */
    public static double dot(INDArray x, INDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT) {
            double ret = BLAS.getInstance().sdot(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.majorStride());
            return ret;

        } else {
            double ret = BLAS.getInstance().ddot(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.majorStride());
            return ret;

        }


    }


    /**
     * Dot product between two complex ndarrays
     *
     * @param x
     * @param y
     * @return
     */
    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT) {
            ComplexFloat f = new ComplexFloat(NativeBlas.cdotc(
                    x.length(),
                    x.data().asFloat(),
                    x.blasOffset(),
                    x.secondaryStride(),
                    y.data().asFloat(),
                    y.blasOffset(),
                    y.secondaryStride()));
            return new ComplexDouble(f.realComponent().doubleValue(), f.imaginaryComponent().doubleValue());

        } else {
            ComplexDouble f = new ComplexDouble(NativeBlas.zdotc(
                    x.length(),
                    x.data().asDouble(),
                    x.blasOffset(),
                    x.secondaryStride(),
                    y.data().asDouble(),
                    y.blasOffset(),
                    y.secondaryStride()));
            return new ComplexDouble(f.realComponent().doubleValue(), f.imaginaryComponent().doubleValue());

        }
    }


    public static INDArray ger(INDArray A, INDArray B, INDArray C, double alpha) {

        DataTypeValidation.assertDouble(A, B, C);
        // = alpha * A * transpose(B) + C
        BLAS.getInstance().dger(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                A.data().asDouble(),        // d_A or x
                A.rows(),   // incx
                B.data().asDouble(),        // dB or y
                B.rows(),   // incy
                C.data().asDouble(),        // dC or A
                C.rows()    // lda
        );


        return C;
    }


    public static INDArray ger(INDArray A, INDArray B, INDArray C, float alpha) {
        DataTypeValidation.assertFloat(A, B, C);

        // = alpha * A * transpose(B) + C
        BLAS.getInstance().sger(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                A.data().asFloat(),        // d_A or x
                A.rows(),   // incx
                B.data().asFloat(),        // dB or y
                B.rows(),   // incy
                C.data().asFloat(),        // dC or A
                C.rows()    // lda
        );


        return C;
    }


    /**
     * Complex dot product
     *
     * @param x
     * @param y
     * @return
     */
    public static IComplexNumber dotu(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT) {
            return new ComplexFloat(NativeBlas.cdotu(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.majorStride()));

        } else {
            return new ComplexDouble(NativeBlas.zdotu(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.majorStride()));

        }

    }

    /**
     * @param alpha
     * @param x
     * @param y
     * @param a
     * @return
     */
    public static IComplexNDArray geru(IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        DataTypeValidation.assertSameDataType(x, y, a);
        if (x.data().dataType() == DataBuffer.FLOAT)
            NativeBlas.cgeru(
                    a.rows(),
                    a.columns(),
                    new ComplexFloat(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue()),
                    x.data().asFloat(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.majorStride(),
                    a.data().asFloat(),
                    a.offset(),
                    a.rows());
        else
            NativeBlas.zgeru(
                    a.rows(),
                    a.columns(),
                    new ComplexDouble(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue()),
                    x.data().asDouble(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.majorStride(),
                    a.data().asDouble(),
                    a.offset(),
                    a.rows());
        return a;
    }

    /**
     * @param x
     * @param y
     * @param a
     * @param alpha
     * @return
     */
    public static IComplexNDArray gerc(IComplexNDArray x, IComplexNDArray y, IComplexNDArray a,
                                       IComplexDouble alpha) {
        DataTypeValidation.assertDouble(x, y, a);
        if (x.data().dataType() == DataBuffer.FLOAT)
            NativeBlas.cgerc(
                    a.rows(),
                    a.columns(),
                    (ComplexFloat) alpha,
                    x.data().asFloat(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.majorStride(),
                    a.data().asFloat(),
                    a.offset(),
                    a.rows());
        else
            NativeBlas.zgerc(
                    a.rows(),
                    a.columns(),
                    (ComplexDouble) alpha,
                    x.data().asDouble(),
                    x.offset(),
                    x.majorStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.majorStride(),
                    a.data().asDouble(),
                    a.offset(),
                    a.rows());
        return a;
    }


    /**
     * Scale a complex ndarray
     *
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray dscal(IComplexDouble alpha, IComplexNDArray x) {
        NativeBlas.zscal(x.length(), (org.jblas.ComplexDouble) alpha, x.data().asDouble(), x.offset(), x.majorStride());
        return x;
    }

    /**
     * Scale a complex ndarray
     *
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray sscal(IComplexFloat alpha, IComplexNDArray x) {
        DataTypeValidation.assertFloat(x);
        NativeBlas.cscal(x.length(), (org.jblas.ComplexFloat) alpha, x.data().asFloat(), x.offset(), x.majorStride());
        return x;
    }


}
