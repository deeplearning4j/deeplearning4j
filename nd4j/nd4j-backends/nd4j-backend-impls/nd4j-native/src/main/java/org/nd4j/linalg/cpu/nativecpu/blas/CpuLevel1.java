/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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

import static org.bytedeco.javacpp.openblas_nolapack.*;


/**
 * @author Adam Gibson
 */
public class CpuLevel1 extends BaseLevel1 {
    private Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();

    @Override
    protected float sdsdot(long N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        return cblas_sdsdot((int) N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected double dsdot(long N, INDArray X, int incX, INDArray Y, int incY) {
        return cblas_dsdot((int) N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected float hdot(long N, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float hdot(long N, DataBuffer X, int offsetX, int incX, DataBuffer Y, int offsetY, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float sdot(long N, INDArray X, int incX, INDArray Y, int incY) {
        if (incX >= 1 && incY >= 1) {
            return cblas_sdot((int) N, (FloatPointer) X.data().addressPointer(), incX,
                            (FloatPointer) Y.data().addressPointer(), incY);
        } else {
            // non-EWS dot variant
            Dot dot = new Dot(X, Y);
            Nd4j.getExecutioner().exec(dot);
            return dot.getFinalResult().floatValue();
        }
    }

    @Override
    protected float sdot(long N, DataBuffer X, int offsetX, int incX, DataBuffer Y, int offsetY, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected double ddot(long N, INDArray X, int incX, INDArray Y, int incY) {
        if (incX >= 1 && incY >= 1) {
            return cblas_ddot((int) N, (DoublePointer) X.data().addressPointer(), incX,
                            (DoublePointer) Y.data().addressPointer(), incY);
        } else {
            // non-EWS dot variant
            Dot dot = new Dot(X, Y);
            Nd4j.getExecutioner().exec(dot);
            return dot.getFinalResult().doubleValue();
        }
    }

    @Override
    protected double ddot(long N, DataBuffer X, int offsetX, int incX, DataBuffer Y, int offsetY, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void cdotu_sub(long N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotu) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void cdotc_sub(long N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotc) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void zdotu_sub(long N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotu) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zdotc_sub(long N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected float snrm2(long N, INDArray X, int incX) {
        return cblas_snrm2((int) N, (FloatPointer) X.data().addressPointer(), incX);

    }

    @Override
    protected float sasum(long N, INDArray X, int incX) {
        return cblas_sasum((int) N, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected float sasum(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected double dnrm2(long N, INDArray X, int incX) {
        return cblas_dnrm2((int) N, (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected double dasum(long N, INDArray X, int incX) {
        return cblas_dasum((int) N, (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected double dasum(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float scnrm2(long N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float scasum(long N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected double dznrm2(long N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected double dzasum(long N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected int isamax(long N, INDArray X, int incX) {
        return (int) cblas_isamax((int) N, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected int isamax(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected int idamax(long N, INDArray X, int incX) {
        return (int) cblas_idamax((int) N, (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected int idamax(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected int icamax(long N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected int izamax(long N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void sswap(long N, INDArray X, int incX, INDArray Y, int incY) {
        cblas_sswap((int) N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void scopy(long N, INDArray X, int incX, INDArray Y, int incY) {
        cblas_scopy((int) N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void scopy(long n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void haxpy(long N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void saxpy(long N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        cblas_saxpy((int) N, alpha, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    public void haxpy(long n, float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void saxpy(long n, float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }


    @Override
    protected void dswap(long N, INDArray X, int incX, INDArray Y, int incY) {
        cblas_dswap((int) N, (DoublePointer) X.data().addressPointer(), incX, (DoublePointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected void dcopy(long N, INDArray X, int incX, INDArray Y, int incY) {
        cblas_dcopy((int) N, (DoublePointer) X.data().addressPointer(), incX, (DoublePointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected void dcopy(long n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void daxpy(long N, double alpha, INDArray X, int incX, INDArray Y, int incY) {
        cblas_daxpy((int) N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Y.data().addressPointer(), incY);

    }

    @Override
    public void daxpy(long n, double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void cswap(long N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void ccopy(long N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void caxpy(long N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zswap(long N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zcopy(long N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zaxpy(long N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
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
    protected void srot(long N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {
        cblas_srot((int) N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(), incY, c,
                        s);
    }

    @Override
    protected void srotm(long N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        cblas_srotm((int) N, (FloatPointer) X.data().addressPointer(), incX, (FloatPointer) Y.data().addressPointer(), incY,
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
    protected void drot(long N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {
        cblas_drot((int) N, (DoublePointer) X.data().addressPointer(), incX, (DoublePointer) Y.data().addressPointer(), incY,
                        c, s);
    }


    @Override
    protected void drotm(long N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        cblas_drotm((int) N, (DoublePointer) X.data().addressPointer(), incX, (DoublePointer) Y.data().addressPointer(), incY,
                        (DoublePointer) P.data().addressPointer());
    }

    @Override
    protected void sscal(long N, float alpha, INDArray X, int incX) {
        cblas_sscal((int) N, alpha, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void dscal(long N, double alpha, INDArray X, int incX) {
        cblas_dscal((int) N, alpha, (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void cscal(long N, IComplexFloat alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void zscal(long N, IComplexDouble alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void csscal(long N, float alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zdscal(long N, double alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected float hasum(long N, INDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float hasum(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException();
    }
}
