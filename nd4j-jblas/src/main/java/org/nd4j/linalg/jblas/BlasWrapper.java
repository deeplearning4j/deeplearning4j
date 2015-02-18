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

package org.nd4j.linalg.jblas;

import org.jblas.JavaBlas;
import org.jblas.NativeBlas;
import org.jblas.exceptions.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.jblas.complex.ComplexDouble;
import org.nd4j.linalg.jblas.complex.ComplexFloat;

import static org.jblas.util.Functions.*;

/**
 * Copy of SimpleBlas to handle offsets implementing
 * an interface for library neutral
 * jblas operations
 *
 * @author Adam Gibson
 */
public class BlasWrapper implements org.nd4j.linalg.factory.BlasWrapper {
    /***************************************************************************
     * BLAS Level 1
     */

    /**
     * Compute x <-> y (swap two matrices)
     */
    @Override
    public INDArray swap(INDArray x, INDArray y) {
        //NativeBlas.dswap(x.length(), x.data(), 0, 1, y.data(), 0, 1);
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT)
            JavaBlas.rswap(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.secondaryStride());
        else
            JavaBlas.rswap(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.secondaryStride());
        return y;
    }

    /**
     * Compute x <- alpha * x (scale a matrix)
     */
    @Override
    public INDArray scal(double alpha, INDArray x) {
        DataTypeValidation.assertDouble(x);
        NativeBlas.dscal(x.length(), alpha, x.data().asDouble(), x.offset(), x.secondaryStride());
        return x;
    }

    /**
     * Compute x <- alpha * x (scale a matrix)
     */
    @Override
    public INDArray scal(float alpha, INDArray x) {
        DataTypeValidation.assertFloat(x);
        NativeBlas.sscal(x.length(), alpha, x.data().asFloat(), x.offset(), x.secondaryStride());
        return x;
    }

    @Override
    public IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) {
        DataTypeValidation.assertFloat(x);
        NativeBlas.cscal(x.length(),
                new ComplexFloat(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue()),
                x.data().asFloat(), x.offset(), x.secondaryStride());
        return x;
    }

    @Override
    public IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) {
        DataTypeValidation.assertDouble(x);
        NativeBlas.zscal(x.length(),
                new ComplexDouble(alpha.realComponent().doubleValue(), alpha.imaginaryComponent().doubleValue()),
                x.data().asDouble(), x.offset(), x.secondaryStride());
        return x;
    }


    /**
     * Compute y <- x (copy a matrix)
     */
    @Override
    public INDArray copy(INDArray x, INDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.DOUBLE)
            JavaBlas.rcopy(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.secondaryStride());
        else
            JavaBlas.rcopy(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.secondaryStride());

        return y;
    }

    @Override
    public IComplexNDArray copy(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.DOUBLE)
            NativeBlas.dcopy(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.secondaryStride());
        else
            NativeBlas.scopy(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.secondaryStride());
        return y;
    }


    /**
     * Compute y <- alpha * x + y (elementwise addition)
     */
    @Override
    public INDArray axpy(double da, INDArray dx, INDArray dy) {
        if (dx.length() != dy.length())
            throw new IllegalArgumentException("Dx and dy must be same length");
        DataTypeValidation.assertDouble(dx, dy);
        JavaBlas.raxpy(
                dx.length(),
                da,
                dx.data().asDouble(),
                dx.offset(),
                dx.secondaryStride(),
                dy.data().asDouble(),
                dy.offset(),
                dy.secondaryStride());

        return dy;
    }

    /**
     * Compute y <- alpha * x + y (elementwise addition)
     */
    @Override
    public INDArray axpy(float da, INDArray dx, INDArray dy) {
        //NativeBlas.daxpy(dx.length(), da, dx.data(), 0, 1, dy.data(), 0, 1);
        assert dx.length() == dy.length() : "Dx length must be the same as dy length";
        DataTypeValidation.assertFloat(dx, dy);
        JavaBlas.raxpy(
                dx.length(),
                da,
                dx.data().asFloat(),
                dx.offset(),
                dx.secondaryStride(),
                dy.data().asFloat(),
                dy.offset(),
                dy.secondaryStride());

        return dy;
    }

    @Override
    public IComplexNDArray axpy(IComplexNumber da, IComplexNDArray dx, IComplexNDArray dy) {
        DataTypeValidation.assertSameDataType(dx, dy);
        if (da instanceof IComplexFloat)
            NativeBlas.caxpy(
                    dx.length(),
                    new org.jblas.ComplexFloat(da.realComponent().floatValue(),
                            da.imaginaryComponent().floatValue()),
                    dx.data().asFloat(),
                    dx.offset(),
                    dx.secondaryStride(),
                    dy.data().asFloat(),
                    dy.offset(),
                    dy.secondaryStride());
        else if (da instanceof IComplexDouble)
            NativeBlas.zaxpy(
                    dx.length(),
                    new org.jblas.ComplexDouble(
                            da.realComponent().doubleValue(),
                            da.imaginaryComponent().doubleValue()
                    ),
                    dx.data().asDouble(),
                    dx.offset(),
                    dx.secondaryStride(),
                    dy.data().asDouble(),
                    dy.offset(),
                    dy.secondaryStride());


        return dy;
    }


    /**
     * Compute x^T * y (dot product)
     */
    @Override
    public double dot(INDArray x, INDArray y) {
        //return NativeBlas.ddot(x.length(), x.data(), 0, 1, y.data(), 0, 1);
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT)
            return JavaBlas.rdot(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.secondaryStride());
        else if (x.data().dataType() == DataBuffer.DOUBLE) {
            return JavaBlas.rdot(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.secondaryStride());
        }

        throw new IllegalStateException("Illegal data type");

    }

    /**
     * Compute x^T * y (dot product)
     */
    @Override
    public IComplexNumber dotc(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT)
            return new ComplexFloat(NativeBlas.cdotc(
                    x.length(),
                    x.data().asFloat(),
                    x.blasOffset(),
                    x.secondaryStride(),
                    y.data().asFloat(),
                    y.blasOffset(),
                    y.secondaryStride()));
        else if (x.data().dataType() == DataBuffer.DOUBLE)
            return new ComplexDouble(
                    NativeBlas.zdotc(
                            x.length(),
                            x.data().asDouble(),
                            x.blasOffset(),
                            x.secondaryStride(),
                            y.data().asDouble(),
                            y.blasOffset(),
                            y.secondaryStride()));
        throw new IllegalStateException("Illegal data type");
    }

    /**
     * Compute x^T * y (dot product)
     */
    @Override
    public IComplexNumber dotu(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x, y);
        if (x.data().dataType() == DataBuffer.FLOAT)
            return new ComplexFloat(NativeBlas.cdotu(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asFloat(),
                    y.offset(),
                    y.secondaryStride()));
        if (x.data().dataType() == DataBuffer.DOUBLE)
            return new ComplexDouble(NativeBlas.zdotu(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.secondaryStride(),
                    y.data().asDouble(),
                    y.offset(),
                    y.secondaryStride()));
        throw new IllegalStateException("Illegal data type");
    }

    /**
     * Compute || x ||_2 (2-norm)
     */
    @Override
    public double nrm2(INDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT)
            return NativeBlas.snrm2(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.secondaryStride());

        if (x.data().dataType() == DataBuffer.DOUBLE)
            return NativeBlas.dnrm2(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.secondaryStride());

        throw new IllegalStateException("Illegal data type");


    }

    @Override
    public double nrm2(IComplexNDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT)
            return NativeBlas.scnrm2(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.secondaryStride());
        else if (x.data().dataType() == DataBuffer.DOUBLE)
            return NativeBlas.dznrm2(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.secondaryStride());

        throw new IllegalStateException("Illegal data type");


    }

    /**
     * Compute || x ||_1 (1-norm, sum of absolute values)
     */
    @Override
    public double asum(INDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT)
            return NativeBlas.sasum(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.secondaryStride());
        if (x.data().dataType() == DataBuffer.DOUBLE)
            return NativeBlas.dasum(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.secondaryStride());
        throw new IllegalStateException("Illegal data type");

    }

    @Override
    public double asum(IComplexNDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT) {
            return NativeBlas.scasum(
                    x.length(),
                    x.data().asFloat(),
                    x.offset() / 2,
                    x.secondaryStride());

        } else if (x.data().dataType() == DataBuffer.DOUBLE) {
            return NativeBlas.dzasum(
                    x.length(),
                    x.data().asDouble(),
                    x.offset() / 2,
                    x.secondaryStride());

        }

        throw new IllegalStateException("Illegal data type");

    }

    /**
     * Compute index of element with largest absolute value (index of absolute
     * value maximum)
     */
    @Override
    public int iamax(INDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT)
            return NativeBlas.isamax(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.secondaryStride()) - 1;
        else if (x.data().dataType() == DataBuffer.DOUBLE) {
            return NativeBlas.idamax(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.secondaryStride()) - 1;

        }
        throw new IllegalStateException("Illegal data type");

    }

    /**
     * Compute index of element with largest absolute value (complex version).
     *
     * @param x matrix
     * @return index of element with largest absolute value.
     */
    @Override
    public int iamax(IComplexNDArray x) {
        if (x.data().dataType() == DataBuffer.FLOAT)
            return NativeBlas.icamax(
                    x.length(),
                    x.data().asFloat(),
                    x.offset(),
                    x.majorStride()) - 1;
        else
            return NativeBlas.izamax(
                    x.length(),
                    x.data().asDouble(),
                    x.offset(),
                    x.majorStride()) - 1;
    }

    /***************************************************************************
     * BLAS Level 2
     */
    /**
     * Compute y <- alpha*op(a)*x + beta * y (general matrix vector
     * multiplication)
     */
    @Override
    public INDArray gemv(double alpha, INDArray a,
                         INDArray x, double beta, INDArray y) {
        DataTypeValidation.assertDouble(a, x, y);
        if (beta == 0.0) {
            for (int j = 0; j < a.columns(); j++) {
                double xj = x.getDouble(j);
                if (xj != 0.0) {
                    for (int i = 0; i < a.rows(); i++) {
                        y.putScalar(i, y.getDouble(i) + a.getDouble(i, j) * xj);
                    }
                }
            }
        } else {
            for (int j = 0; j < a.columns(); j++) {
                double byj = beta * y.data().getDouble(j);
                double xj = x.getFloat(j);
                for (int i = 0; i < a.rows(); i++) {
                    y.putScalar(j, a.getDouble(i, j) * xj + byj);
                }
            }
        }

        return y;
    }

    /**
     * Compute y <- alpha*op(a)*x + beta * y (general matrix vector
     * multiplication)
     */
    @Override
    public INDArray gemv(float alpha, INDArray a,
                         INDArray x, float beta, INDArray y) {
        DataTypeValidation.assertFloat(a, x, y);
        if (beta == 0.0) {
            for (int j = 0; j < a.columns(); j++) {
                double xj = x.getFloat(j);
                if (xj != 0.0) {
                    for (int i = 0; i < a.rows(); i++) {
                        y.putScalar(i, y.getFloat(i) + a.getFloat(i, j) * xj);
                    }
                }
            }
        } else {
            for (int j = 0; j < a.columns(); j++) {
                double byj = beta * y.data().getDouble(j);
                double xj = x.getFloat(j);
                for (int i = 0; i < a.rows(); i++) {
                    y.putScalar(j, a.getFloat(i, j) * xj + byj);
                }
            }
        }

        return y;
    }


    /**
     * Compute A <- alpha * x * y^T + A (general rank-1 update)
     */
    @Override
    public INDArray ger(double alpha, INDArray x,
                        INDArray y, INDArray a) {
        DataTypeValidation.assertDouble(x, y, a);
        NativeBlas.dger(
                a.rows(),
                a.columns(),
                alpha,
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
     * Compute A <- alpha * x * y^T + A (general rank-1 update)
     */
    @Override
    public INDArray ger(float alpha, INDArray x,
                        INDArray y, INDArray a) {
        DataTypeValidation.assertFloat(x, y, a);
        NativeBlas.sger(
                a.rows(),
                a.columns(),
                alpha,
                x.data().asFloat(),
                x.offset(),
                x.majorStride(),
                y.data().asFloat(),
                y.offset(),
                y.majorStride(),
                a.data().asFloat(),
                a.offset(),
                a.rows());
        return a;
    }

    @Override
    public IComplexNDArray gemv(IComplexDouble alpha, IComplexNDArray a, IComplexNDArray x, IComplexDouble beta, IComplexNDArray y) {
        DataTypeValidation.assertDouble(a, x, y);
        if (y.isScalar())
            return y.putScalar(0, dotc(a, x));
        NativeBlas.zgemv(
                'N',
                a.rows(),
                a.columns(),
                (ComplexDouble) alpha,
                a.data().asDouble(),
                a.blasOffset(),
                a.rows(),
                x.data().asDouble(),
                x.offset(),
                x.secondaryStride(),
                (ComplexDouble) beta,
                y.data().asDouble(),
                y.blasOffset(),
                y.secondaryStride()
        );
        return y;
    }

    @Override
    public IComplexNDArray gemv(IComplexFloat alpha, IComplexNDArray a, IComplexNDArray x, IComplexFloat beta, IComplexNDArray y) {
        DataTypeValidation.assertDouble(a, x, y);
        NativeBlas.cgemv(
                'N',
                a.rows(),
                a.columns(),
                (ComplexFloat) alpha,
                a.data().asFloat(),
                a.blasOffset(),
                a.rows(),
                x.data().asFloat(),
                x.offset(),
                x.secondaryStride(),
                (ComplexFloat) beta,
                y.data().asFloat(),
                y.blasOffset(),
                y.secondaryStride()
        );
        return y;
    }

    /**
     * Compute A <- alpha * x * y^T + A (general rank-1 update)
     *
     * @param alpha
     * @param x
     * @param y
     * @param a
     */
    @Override
    public IComplexNDArray geru(IComplexDouble alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        DataTypeValidation.assertDouble(x, y, a);
        NativeBlas.zgeru(
                a.rows(),
                a.columns(),
                new ComplexDouble(alpha.realComponent().doubleValue(), alpha.imaginaryComponent().doubleValue()),
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
     * Compute A <- alpha * x * y^T + A (general rank-1 update)
     *
     * @param alpha
     * @param x
     * @param y
     * @param a
     */
    @Override
    public IComplexNDArray geru(IComplexFloat alpha, IComplexNDArray x, IComplexNDArray y, IComplexNDArray a) {
        DataTypeValidation.assertFloat(x, y, a);
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
        return a;
    }


    /**
     * Compute A <- alpha * x * y^H + A (general rank-1 update)
     */
    @Override
    public IComplexNDArray gerc(IComplexDouble alpha, IComplexNDArray x,
                                IComplexNDArray y, IComplexNDArray a) {
        DataTypeValidation.assertDouble(x, y, a);
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
     * Compute A <- alpha * x * y^H + A (general rank-1 update)
     */
    @Override
    public IComplexNDArray gerc(IComplexFloat alpha, IComplexNDArray x,
                                IComplexNDArray y, IComplexNDArray a) {
        DataTypeValidation.assertFloat(x, y, a);
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
        return a;
    }

    /***************************************************************************
     * BLAS Level 3
     */


    /**
     * Compute c <- a*b + beta * c (general matrix matrix
     * multiplication)
     */
    @Override
    public INDArray gemm(double alpha, INDArray a,
                         INDArray b, double beta, INDArray c) {
        DataTypeValidation.assertDouble(a, b, c);
        if (a.shape().length > 2) {
            for (int i = 0; i < a.slices(); i++) {
                c.putSlice(i, a.slice(i).mmul(b.slice(i)));
            }

            return c;
        }

        NativeBlas.dgemm(
                'N',
                'N',
                c.rows(),
                c.columns(),
                a.columns(),
                alpha,
                a.data().asDouble(),
                a.offset(),
                a.rows()
                , b.data().asDouble(),
                b.offset(),
                b.rows(),
                beta,
                c.data().asDouble(),
                c.offset(),
                c.rows());

        return c;
    }

    /**
     * Compute c <- a*b + beta * c (general matrix matrix
     * multiplication)
     */
    @Override
    public INDArray gemm(float alpha, INDArray a,
                         INDArray b, float beta, INDArray c) {
        DataTypeValidation.assertFloat(a, b, c);
        if (a.shape().length > 2) {
            for (int i = 0; i < a.slices(); i++) {
                c.putSlice(i, a.slice(i).mmul(b.slice(i)));
            }

            return c;
        }


        NativeBlas.sgemm(
                'N',
                'N',
                c.rows(),
                c.columns(),
                a.columns(),
                alpha,
                a.data().asFloat(),
                a.offset(),
                a.rows()
                , b.data().asFloat(),
                b.offset(),
                b.rows(),
                beta,
                c.data().asFloat(),
                c.offset(),
                c.rows());

        return c;
    }


    @Override
    public IComplexNDArray gemm(IComplexNumber alpha, IComplexNDArray a, IComplexNDArray b, IComplexNumber beta, IComplexNDArray c) {
        DataTypeValidation.assertSameDataType(a, b, c);
        if (a.data().dataType() == DataBuffer.FLOAT)
            NativeBlas.cgemm(
                    'N',
                    'N',
                    c.rows(),
                    c.columns(),
                    a.columns(),
                    new ComplexFloat(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue()),
                    a.data().asFloat(),
                    a.blasOffset(),
                    a.rows(),
                    b.data().asFloat(),
                    b.blasOffset(),
                    b.rows(),
                    new ComplexFloat(beta.realComponent().floatValue(), beta.imaginaryComponent().floatValue())
                    , c.data().asFloat(),
                    c.blasOffset(),
                    c.rows());
        else
            NativeBlas.zgemm(
                    'N',
                    'N',
                    c.rows(),
                    c.columns(),
                    a.columns(),
                    new ComplexDouble(alpha.realComponent().floatValue(), alpha.imaginaryComponent().floatValue()),
                    a.data().asDouble(),
                    a.blasOffset(),
                    a.rows(),
                    b.data().asDouble(),
                    b.blasOffset(),
                    b.rows(),
                    new ComplexDouble(beta.realComponent().floatValue(), beta.imaginaryComponent().floatValue())
                    , c.data().asDouble(),
                    c.blasOffset(),
                    c.rows());
        return c;

    }


    /**
     * ************************************************************************
     * LAPACK
     */

    @Override
    public INDArray gesv(INDArray a, int[] ipiv,
                         INDArray b) {
        DataTypeValidation.assertSameDataType(a, b);
        int info = -1;
        if (a.data().dataType() == DataBuffer.FLOAT) {
            info = NativeBlas.sgesv(
                    a.rows(),
                    b.columns(),
                    a.data().asFloat(),
                    a.offset(),
                    a.rows(),
                    ipiv,
                    0,
                    b.data().asFloat(),
                    b.offset(),
                    b.rows());
        } else if (a.data().dataType() == DataBuffer.DOUBLE) {
            info = NativeBlas.dgesv(
                    a.rows(),
                    b.columns(),
                    a.data().asDouble(),
                    a.offset(),
                    a.rows(),
                    ipiv,
                    0,
                    b.data().asDouble(),
                    b.offset(),
                    b.rows());
        }
        checkInfo("DGESV", info);

        if (info > 0)
            throw new LapackException("DGESV",
                    "Linear equation cannot be solved because the matrix was singular.");

        return b;
    }

//STOP

    public void checkInfo(String name, int info) {
        if (info < -1)
            throw new LapackArgumentException(name, info);
    }
//START

    @Override
    public INDArray sysv(char uplo, INDArray a, int[] ipiv,
                         INDArray b) {
        DataTypeValidation.assertSameDataType(a, b);
        int info = -1;
        if (a.data().dataType() == DataBuffer.FLOAT) {
            info = NativeBlas.ssysv(
                    uplo,
                    a.rows(),
                    b.columns(),
                    a.data().asFloat(),
                    a.offset(),
                    a.rows(),
                    ipiv,
                    0,
                    b.data().asFloat(),
                    b.offset(),
                    b.rows());
        } else if (a.data().dataType() == DataBuffer.DOUBLE) {
            info = NativeBlas.dsysv(
                    uplo,
                    a.rows(),
                    b.columns(),
                    a.data().asDouble(),
                    a.offset(),
                    a.rows(),
                    ipiv,
                    0,
                    b.data().asDouble(),
                    b.offset(),
                    b.rows());

        }

        checkInfo("SYSV", info);

        if (info > 0)
            throw new LapackSingularityException("SYV",
                    "Linear equation cannot be solved because the matrix was singular.");

        return b;
    }

    @Override
    public int syev(char jobz, char uplo, INDArray a, INDArray w) {
        int info = -1;
        DataTypeValidation.assertSameDataType(a, w);
        if (a.data().dataType() == DataBuffer.FLOAT) {
            info = NativeBlas.ssyev(
                    jobz,
                    uplo,
                    a.rows(),
                    a.data().asFloat(),
                    a.offset(),
                    a.rows(),
                    w.data().asFloat(),
                    w.offset());

        } else {
            info = NativeBlas.dsyev(
                    jobz,
                    uplo,
                    a.rows(),
                    a.data().asDouble(),
                    a.offset(),
                    a.rows(),
                    w.data().asDouble(),
                    w.offset());

        }


        if (info > 0)
            throw new LapackConvergenceException("SYEV",
                    "Eigenvalues could not be computed " + info
                            + " off-diagonal elements did not converge");

        return info;
    }


    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a,
                     double vl, double vu, int il, int iu, double abstol,
                     INDArray w, INDArray z) {
        DataTypeValidation.assertDouble(a, w, z);
        int n = a.rows();
        int[] iwork = new int[5 * n];
        int[] ifail = new int[n];
        int[] m = new int[1];
        int info;

        info = NativeBlas.dsyevx(jobz, range, uplo, n, a.data().asDouble(), a.offset(), a.rows(), vl, vu, il,
                iu, abstol, m, 0, w.data().asDouble(), w.offset(), z.data().asDouble(), z.offset(), z.rows(), iwork, 0, ifail, 0);

        if (info > 0) {
            StringBuilder msg = new StringBuilder();
            msg
                    .append("Not all eigenvalues converged. Non-converging eigenvalues were: ");
            for (int i = 0; i < info; i++) {
                if (i > 0)
                    msg.append(", ");
                msg.append(ifail[i]);
            }
            msg.append(".");
            throw new LapackConvergenceException("SYEVX", msg.toString());
        }

        return info;

    }

    @Override
    public int syevx(char jobz, char range, char uplo, INDArray a,
                     float vl, float vu, int il, int iu, float abstol,
                     INDArray w, INDArray z) {
        DataTypeValidation.assertFloat(a, w, z);
        int n = a.rows();
        int[] iwork = new int[5 * n];
        int[] ifail = new int[n];
        int[] m = new int[1];
        int info;

        info = NativeBlas.ssyevx(jobz, range, uplo, n, a.data().asFloat(), a.offset(), a.rows(), vl, vu, il,
                iu, abstol, m, 0, w.data().asFloat(), w.offset(), z.data().asFloat(), z.offset(), z.rows(), iwork, 0, ifail, 0);

        if (info > 0) {
            StringBuilder msg = new StringBuilder();
            msg
                    .append("Not all eigenvalues converged. Non-converging eigenvalues were: ");
            for (int i = 0; i < info; i++) {
                if (i > 0)
                    msg.append(", ");
                msg.append(ifail[i]);
            }
            msg.append(".");
            throw new LapackConvergenceException("SYEVX", msg.toString());
        }

        return info;
    }

    public int syevd(char jobz, char uplo, INDArray A,
                     INDArray w) {
        int n = A.rows();
        DataTypeValidation.assertSameDataType(A, w);
        int info = -1;
        if (A.data().dataType() == DataBuffer.FLOAT) {
            info = NativeBlas.ssyevd(
                    jobz,
                    uplo,
                    n,
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    w.data().asFloat(),
                    w.offset());

        } else if (A.data().dataType() == DataBuffer.DOUBLE) {
            info = NativeBlas.dsyevd(
                    jobz,
                    uplo,
                    n,
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    w.data().asDouble(),
                    w.offset());

        }

        if (info > 0)
            throw new LapackConvergenceException("SYEVD", "Not all eigenvalues converged.");

        return info;
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a, double vl, double vu, int il, int iu, double abstol, INDArray w, INDArray z, int[] isuppz) {
        return 0;
    }

    @Override
    public int syevr(char jobz, char range, char uplo, INDArray a,
                     float vl, float vu, int il, int iu, float abstol,
                     INDArray w, INDArray z, int[] isuppz) {
        int n = a.rows();
        int[] m = new int[1];
        DataTypeValidation.assertFloat(a, w, z);
        int info = -1;
        if (w.data().dataType() == DataBuffer.FLOAT) {
            info = NativeBlas.ssyevr(
                    jobz,
                    range,
                    uplo,
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
                    0,
                    w.data().asFloat(),
                    w.offset(),
                    z.data().asFloat(),
                    z.offset(),
                    z.rows(),
                    isuppz,
                    0);

        } else if (w.data().dataType() == DataBuffer.DOUBLE) {
            info = NativeBlas.dsyevr(
                    jobz,
                    range,
                    uplo,
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
                    0,
                    w.data().asDouble(),
                    w.offset(),
                    z.data().asDouble(),
                    z.offset(),
                    z.rows(),
                    isuppz,
                    0);

        }


        checkInfo("SYEVR", info);

        return info;
    }

    @Override
    public void posv(char uplo, INDArray A, INDArray B) {
        int n = A.rows();
        int nrhs = B.columns();
        int info = -1;
        DataTypeValidation.assertSameDataType(A, B);
        if (A.data().dataType() == DataBuffer.FLOAT)
            info = NativeBlas.sposv(
                    uplo,
                    n,
                    nrhs,
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    B.data().asFloat(),
                    B.offset(),
                    B.rows());
        else if (A.data().dataType() == DataBuffer.DOUBLE) {
            info = NativeBlas.dposv(
                    uplo,
                    n,
                    nrhs,
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    B.data().asDouble(),
                    B.offset(),
                    B.rows());
        }
        checkInfo("DPOSV", info);
        if (info > 0)
            throw new LapackArgumentException("DPOSV",
                    "Leading minor of order i of A is not positive definite.");
    }

    @Override
    public int geev(char jobvl, char jobvr, INDArray A,
                    INDArray WR, INDArray WI, INDArray VL, INDArray VR) {
        int n = A.rows();
        assert WR.length() == n;
        assert WI.length() == n;
        assert VL.columns() == n;
        assert VR.columns() == n;
        int ldvl = VL.rows();
        int ldvr = VR.rows();
        DataTypeValidation.assertSameDataType(A, WR, WI, VL, VR);
        if (Character.toLowerCase(jobvl) == 'v')
            assert ldvl >= n;

        if (Character.toLowerCase(jobvr) == 'r')
            assert ldvr >= n;
        int info = -1;
        if (A.data().dataType() == DataBuffer.FLOAT)
            info = NativeBlas.sgeev(
                    jobvl,
                    jobvr,
                    A.rows(),
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    WR.data().asFloat(),
                    WR.offset(),
                    WI.data().asFloat(),
                    WI.offset(),
                    VL.data().asFloat(),
                    VL.offset(),
                    ldvl,
                    VR.data().asFloat(),
                    VR.offset(),
                    ldvr);
        else if (A.data().dataType() == DataBuffer.DOUBLE)
            info = NativeBlas.dgeev(
                    jobvl,
                    jobvr,
                    A.rows(),
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    WR.data().asDouble(),
                    WR.offset(),
                    WI.data().asDouble(),
                    WI.offset(),
                    VL.data().asDouble(),
                    VL.offset(),
                    ldvl,
                    VR.data().asDouble(),
                    VR.offset(),
                    ldvr);
        if (info > 0)
            throw new LapackConvergenceException("DGEEV", "First " + info + " eigenvalues have not converged.");
        return info;
    }

    @Override
    public int sygvd(int itype, char jobz, char uplo, INDArray A, INDArray B, INDArray W) {
        int info = -1;
        DataTypeValidation.assertSameDataType(A, B, W);
        if (A.data().dataType() == DataBuffer.DOUBLE) {
            info = NativeBlas.dsygvd(
                    itype,
                    jobz,
                    uplo,
                    A.rows(),
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    B.data().asDouble(),
                    B.offset(),
                    B.rows(),
                    W.data().asDouble(),
                    W.offset());

        } else {
            info = NativeBlas.ssygvd(
                    itype,
                    jobz,
                    uplo,
                    A.rows(),
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    B.data().asFloat(),
                    B.offset(),
                    B.rows(),
                    W.data().asFloat(),
                    W.offset());

        }
        if (info == 0)
            return 0;
        else {
            if (info < 0)
                throw new LapackArgumentException("DSYGVD", -info);
            if (info <= A.rows() && jobz == 'N')
                throw new LapackConvergenceException("DSYGVD", info + " off-diagonal elements did not converge to 0.");
            if (info <= A.rows() && jobz == 'V')
                throw new LapackException("DSYGVD", "Failed to compute an eigenvalue while working on a sub-matrix  " + info + ".");
            else
                throw new LapackException("DSYGVD", "The leading minor of order " + (info - A.rows()) + " of B is not positive definite.");
        }
    }

    /**
     * Generalized Least Squares via *GELSD.
     * <p/>
     * Note that B must be padded to contain the solution matrix. This occurs when A has fewer rows
     * than columns.
     * <p/>
     * For example: in A * X = B, A is (m,n), X is (n,k) and B is (m,k). Now if m < n, since B is overwritten to contain
     * the solution (in classical LAPACK style), B needs to be padded to be an (n,k) matrix.
     * <p/>
     * Likewise, if m > n, the solution consists only of the first n rows of B.
     *
     * @param A an (m,n) matrix
     * @param B an (max(m,n), k) matrix (well, at least)
     */
    @Override
    public void gelsd(INDArray A, INDArray B) {
        int m = A.rows();
        int n = A.columns();
        int nrhs = B.columns();
        int minmn = min(m, n);
        int maxmn = max(m, n);
        DataTypeValidation.assertSameDataType(A, B);
        if (B.rows() < maxmn) {
            throw new SizeException("Result matrix B must be padded to contain the solution matrix X!");
        }

        int smlsiz = NativeBlas.ilaenv(9, "DGELSD", "", m, n, nrhs, 0);
        int nlvl = max(0, (int) log2(minmn / (smlsiz + 1)) + 1);

        int[] iwork = new int[3 * minmn * nlvl + 11 * minmn];
        int[] rank = new int[1];

        if (A.data().dataType() == DataBuffer.FLOAT) {
            float[] s = new float[minmn];
            int info = NativeBlas.sgelsd(
                    m,
                    n,
                    nrhs,
                    A.data().asFloat(),
                    A.offset(),
                    m,
                    B.data().asFloat(),
                    B.offset(),
                    B.rows(),
                    s,
                    0,
                    -1,
                    rank,
                    0,
                    iwork,
                    0);
            if (info == 0) {
                return;
            } else if (info < 0) {
                throw new LapackArgumentException("DGESD", -info);
            } else if (info > 0) {
                throw new LapackConvergenceException("DGESD", info + " off-diagonal elements of an intermediat bidiagonal form did not converge to 0.");
            }
        } else {
            double[] s = new double[minmn];
            int info = NativeBlas.dgelsd(
                    m,
                    n,
                    nrhs,
                    A.data().asDouble(),
                    A.offset(),
                    m,
                    B.data().asDouble(),
                    B.offset(),
                    B.rows(),
                    s,
                    0,
                    -1,
                    rank,
                    0,
                    iwork,
                    0);
            if (info == 0) {
                return;
            } else if (info < 0) {
                throw new LapackArgumentException("DGESD", -info);
            } else if (info > 0) {
                throw new LapackConvergenceException("DGESD", info + " off-diagonal elements of an intermediat bidiagonal form did not converge to 0.");
            }
        }

    }

    @Override
    public void geqrf(INDArray A, INDArray tau) {
        DataTypeValidation.assertSameDataType(A, tau);
        if (A.data().dataType() == DataBuffer.FLOAT) {
            int info = NativeBlas.sgeqrf(
                    A.rows(),
                    A.columns(),
                    A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    tau.data().asFloat(),
                    tau.offset());
            checkInfo("GEQRF", info);
        } else {
            int info = NativeBlas.dgeqrf(
                    A.rows(),
                    A.columns(),
                    A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    tau.data().asDouble(),
                    tau.offset());
            checkInfo("GEQRF", info);
        }

    }

    @Override
    public void ormqr(char side, char trans, INDArray A, INDArray tau, INDArray C) {
        int k = tau.length();
        DataTypeValidation.assertSameDataType(A, tau, C);
        if (A.data().dataType() == DataBuffer.FLOAT) {
            int info = NativeBlas.sormqr(
                    side,
                    trans,
                    C.rows(),
                    C.columns(),
                    k, A.data().asFloat(),
                    A.offset(),
                    A.rows(),
                    tau.data().asFloat(),
                    0,
                    C.data().asFloat(),
                    0,
                    C.rows());
            checkInfo("ORMQR", info);
        } else {
            int info = NativeBlas.dormqr(
                    side,
                    trans,
                    C.rows(),
                    C.columns(),
                    k, A.data().asDouble(),
                    A.offset(),
                    A.rows(),
                    tau.data().asDouble(),
                    0,
                    C.data().asDouble(),
                    0,
                    C.rows());
            checkInfo("ORMQR", info);
        }
    }

    @Override
    public void dcopy(int n, float[] dx, int dxIdx, int incx, float[] dy, int dyIdx, int incy) {
        NativeBlas.scopy(n, dx, dxIdx, incx, dy, dyIdx, incy);
    }

    /**
     * Abstraction over saxpy
     *
     * @param alpha the alpha to scale by
     * @param x     the ndarray to use
     * @param y     the ndarray to use
     */
    @Override
    public void saxpy(double alpha, INDArray x, INDArray y) {
        DataTypeValidation.assertDouble(x, y);
        JavaBlas.raxpy(
                x.length(),
                alpha,
                x.data().asDouble(),
                x.offset(),
                x.secondaryStride(),
                y.data().asDouble(),
                y.offset(),
                y.secondaryStride());

    }

    /**
     * Abstraction over saxpy
     *
     * @param alpha the alpha to scale by
     * @param x     the ndarray to use
     * @param y     the ndarray to use
     */
    @Override
    public void saxpy(float alpha, INDArray x, INDArray y) {
        DataTypeValidation.assertFloat(x, y);
        JavaBlas.raxpy(
                x.length(),
                alpha,
                x.data().asFloat(),
                x.offset(),
                x.secondaryStride(),
                y.data().asFloat(),
                y.offset(),
                y.secondaryStride());

    }

}
