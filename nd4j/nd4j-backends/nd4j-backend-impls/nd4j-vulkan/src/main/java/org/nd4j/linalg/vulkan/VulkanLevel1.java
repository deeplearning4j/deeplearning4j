/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.linalg.vulkan;

import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Device-backed Level-1 BLAS compatibility operations.
 *
 * <p>These methods are expressed in ND4J operations deliberately. They therefore
 * use the active backend's descriptor/emitter path instead of silently reaching
 * for a host BLAS installation.</p>
 */
final class VulkanLevel1 implements Level1 {
    private static INDArray view(DataBuffer data, long n, int offset, int increment) {
        return Nd4j.create(data, new long[]{n}, new long[]{increment}, offset, 'c');
    }

    @Override
    public double dot(long n, double alpha, INDArray x, INDArray y) {
        return x.mul(y).sumNumber().doubleValue();
    }

    @Override
    public double dot(long n, DataBuffer x, int offsetX, int incrX,
                      DataBuffer y, int offsetY, int incrY) {
        return dot(n, 1.0, view(x, n, offsetX, incrX), view(y, n, offsetY, incrY));
    }

    @Override
    public double nrm2(INDArray arr) {
        return arr.norm2Number().doubleValue();
    }

    @Override
    public double asum(INDArray arr) {
        return Transforms.abs(arr, false).sumNumber().doubleValue();
    }

    @Override
    public double asum(long n, DataBuffer x, int offsetX, int incrX) {
        return asum(view(x, n, offsetX, incrX));
    }

    @Override
    public int iamax(INDArray arr) {
        return Nd4j.argMax(Transforms.abs(arr, false)).getInt(0);
    }

    @Override
    public int iamax(long n, INDArray arr, int stride) {
        return iamax(Nd4j.create(arr.data(), new long[]{n}, new long[]{stride}, arr.offset(), arr.ordering()));
    }

    @Override
    public int iamax(long n, DataBuffer x, int offsetX, int incrX) {
        return iamax(view(x, n, offsetX, incrX));
    }

    @Override
    public int iamin(INDArray arr) {
        return Nd4j.argMin(Transforms.abs(arr, false)).getInt(0);
    }

    @Override
    public void swap(INDArray x, INDArray y) {
        INDArray tmp = x.dup();
        x.assign(y);
        y.assign(tmp);
    }

    @Override
    public void copy(INDArray x, INDArray y) {
        y.assign(x);
    }

    @Override
    public void copy(long n, DataBuffer x, int offsetX, int incrX,
                     DataBuffer y, int offsetY, int incrY) {
        copy(view(x, n, offsetX, incrX), view(y, n, offsetY, incrY));
    }

    @Override
    public void axpy(long n, double alpha, INDArray x, INDArray y) {
        y.assign(x.mul(alpha).add(y));
    }

    @Override
    public void axpy(long n, double alpha, DataBuffer x, int offsetX, int incrX,
                     DataBuffer y, int offsetY, int incrY) {
        axpy(n, alpha, view(x, n, offsetX, incrX), view(y, n, offsetY, incrY));
    }

    @Override
    public void rotg(INDArray a, INDArray b, INDArray c, INDArray s) {
        double av = a.getDouble(0);
        double bv = b.getDouble(0);
        double r = Math.hypot(av, bv);
        double cv = r == 0.0 ? 1.0 : av / r;
        double sv = r == 0.0 ? 0.0 : bv / r;
        a.putScalar(0, r);
        b.putScalar(0, 0.0);
        c.putScalar(0, cv);
        s.putScalar(0, sv);
    }

    @Override
    public void rot(long n, INDArray x, INDArray y, double c, double s) {
        INDArray xv = x.dup();
        INDArray yv = y.dup();
        x.assign(xv.mul(c).add(yv.mul(s)));
        y.assign(yv.mul(c).sub(xv.mul(s)));
    }

    @Override
    public void rotmg(INDArray d1, INDArray d2, INDArray b1, double b2, INDArray p) {
        // Netlib DROTMG/ SROTMG reference algorithm. The scalar parameters are
        // intentionally read back once; all vector updates remain device ops.
        final double gam = 4096.0;
        final double gamsq = gam * gam;
        final double rgamsq = 1.0 / gamsq;
        double d1v = d1.getDouble(0);
        double d2v = d2.getDouble(0);
        double b1v = b1.getDouble(0);
        double flag;
        double h11 = 0.0;
        double h12 = 0.0;
        double h21 = 0.0;
        double h22 = 0.0;
        if (d1v < 0.0) {
            flag = -1.0;
            d1v = d2v = 0.0;
            b1v = 0.0;
        } else {
            double p2 = d2v * b2;
            if (p2 == 0.0) {
                p.putScalar(0, -2.0);
                p.putScalar(1, 0.0);
                p.putScalar(2, 0.0);
                p.putScalar(3, 0.0);
                p.putScalar(4, 0.0);
                return;
            }
            double p1 = d1v * b1v;
            double q2 = p2 * b2;
            double q1 = p1 * b1v;
            if (Math.abs(q1) > Math.abs(q2)) {
                h21 = -b2 / b1v;
                h12 = p2 / p1;
                double u = 1.0 - h12 * h21;
                flag = 0.0;
                d1v /= u;
                d2v /= u;
                b1v *= u;
            } else if (q2 < 0.0) {
                flag = -1.0;
                d1v = d2v = 0.0;
                b1v = 0.0;
            } else {
                flag = 1.0;
                h11 = p1 / p2;
                h22 = -b1v / b2;
                double u = 1.0 + h11 * h22;
                double old = d1v;
                d1v = d2v / u;
                d2v = old / u;
                b1v = b2 * u;
            }
        }
        if (d1v != 0.0 && d2v != 0.0) {
            while (d1v <= rgamsq || d1v >= gamsq) {
                if (flag == 0.0) { h11 /= gam; h12 /= gam; }
                else if (flag == 1.0) { h21 /= gam; h22 /= gam; }
                d1v = d1v <= rgamsq ? d1v * gamsq : d1v / gamsq;
                b1v = b1v <= rgamsq ? b1v * gamsq : b1v / gamsq;
            }
            while (d2v <= rgamsq || d2v >= gamsq) {
                if (flag == 0.0) { h11 *= gam; h12 *= gam; }
                else if (flag == 1.0) { h21 *= gam; h22 *= gam; }
                d2v = d2v <= rgamsq ? d2v * gamsq : d2v / gamsq;
            }
        }
        p.putScalar(0, flag);
        p.putScalar(1, h11);
        p.putScalar(2, h21);
        p.putScalar(3, h12);
        p.putScalar(4, h22);
        d1.putScalar(0, d1v);
        d2.putScalar(0, d2v);
        b1.putScalar(0, b1v);
    }

    @Override
    public void scal(long n, double alpha, INDArray x) {
        x.muli(alpha);
    }

    @Override
    public boolean supportsDataBufferL1Ops() {
        return false;
    }
}
