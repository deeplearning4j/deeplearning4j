package org.nd4j.autodiff;

import java.util.Random;

public class DoubleDoubleComplexFactory
        implements AbstractFactory<DoubleDoubleComplex> {

    private static final DoubleDoubleComplexFactory m_INSTANCE = new DoubleDoubleComplexFactory();
    private static final DoubleDoubleComplex ZERO = new DoubleDoubleComplex(0.0, 0.0);
    private static final DoubleDoubleComplex ONE = new DoubleDoubleComplex(1.0, 0.0);

    private Random randomGenerator = new Random();
    private DoubleDoubleRealFactory realFactory;

    private DoubleDoubleComplexFactory() {
        realFactory = DoubleDoubleRealFactory.instance();
    }

    public static DoubleDoubleComplexFactory instance() {
        return m_INSTANCE;
    }

    @Override
    public DoubleDoubleComplex zero() {
        return ZERO;
    }

    @Override
    public DoubleDoubleComplex one() {
        return ONE;
    }

    @Override
    public DoubleDoubleComplex val(double value) {
        return new DoubleDoubleComplex(value);
    }

    @Override
    public DoubleDoubleComplex abs(DoubleDoubleComplex i_x) {
        return i_x.abs();
    }

    @Override
    public DoubleDoubleComplex min(DoubleDoubleComplex i_x, DoubleDoubleComplex i_y) {
        return new DoubleDoubleComplex(realFactory.min(i_x.re(), i_y.re()));
    }

    @Override
    public DoubleDoubleComplex max(DoubleDoubleComplex i_x, DoubleDoubleComplex i_y) {
        return new DoubleDoubleComplex(realFactory.max(i_x.re(), i_y.re()));
    }

    @Override
    public DoubleDoubleComplex cos(DoubleDoubleComplex value) {
        return value.cos();
    }

    @Override
    public DoubleDoubleComplex acos(DoubleDoubleComplex value) {
        return value.acos();
    }

    @Override
    public DoubleDoubleComplex cosh(DoubleDoubleComplex value) {
        return value.cosh();
    }

    @Override
    public DoubleDoubleComplex acosh(DoubleDoubleComplex value) {
        return value.acosh();
    }

    @Override
    public DoubleDoubleComplex sin(DoubleDoubleComplex value) {
        return value.sin();
    }

    @Override
    public DoubleDoubleComplex asin(DoubleDoubleComplex value) {
        return value.asin();
    }

    @Override
    public DoubleDoubleComplex sinh(DoubleDoubleComplex value) {
        return value.sinh();
    }

    @Override
    public DoubleDoubleComplex asinh(DoubleDoubleComplex value) {
        return value.asinh();
    }

    @Override
    public DoubleDoubleComplex tan(DoubleDoubleComplex value) {
        return value.tan();
    }

    @Override
    public DoubleDoubleComplex atan(DoubleDoubleComplex value) {
        return value.atan();
    }

    @Override
    public DoubleDoubleComplex atan2(DoubleDoubleComplex i_x, DoubleDoubleComplex i_y) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DoubleDoubleComplex tanh(DoubleDoubleComplex value) {
        return value.tanh();
    }

    @Override
    public DoubleDoubleComplex atanh(DoubleDoubleComplex value) {
        return value.atanh();
    }

    @Override
    public DoubleDoubleComplex exp(DoubleDoubleComplex value) {
        return value.exp();
    }

    @Override
    public DoubleDoubleComplex log(DoubleDoubleComplex value) {
        return value.log();
    }

    @Override
    public DoubleDoubleComplex log10(DoubleDoubleComplex value) {
        return value.log10();
    }

    @Override
    public DoubleDoubleComplex flat(DoubleDoubleComplex i_x) {
        return new DoubleDoubleComplex(realFactory.flat(i_x.re()));
    }

    @Override
    public DoubleDoubleComplex mc(DoubleDoubleComplex i_x, DoubleDoubleComplex i_y) {
        return new DoubleDoubleComplex(realFactory.mc(i_x.re(), i_y.re()));
    }

    @Override
    public DoubleDoubleComplex rand(DoubleDoubleComplex i_x) {
        return new DoubleDoubleComplex(new Random((long) i_x.getReal()).nextDouble());
    }

    @Override
    public DoubleDoubleComplex random(DoubleDoubleComplex i_x) {
        return new DoubleDoubleComplex(new Random((long) i_x.getReal()).nextDouble());
    }

    @Override
    public DoubleDoubleComplex gauss(DoubleDoubleComplex i_x) {
        return new DoubleDoubleComplex(realFactory.gauss(i_x.re()));
    }

    @Override
    public DoubleDoubleComplex sgn(DoubleDoubleComplex i_x) {
        return new DoubleDoubleComplex(realFactory.sgn(i_x.re()));
    }

    @Override
    public DoubleDoubleComplex ifx(DoubleDoubleComplex i_x, DoubleDoubleComplex i_y,
            DoubleDoubleComplex i_z) {
        return new DoubleDoubleComplex(realFactory.ifx(i_x.re(), i_y.re(), i_z.re()));
    }

    @Override
    public DoubleDoubleComplex buf(DoubleDoubleComplex i_x) {
        return new DoubleDoubleComplex(realFactory.buf(i_x.re()));
    }

    @Override
    public DoubleDoubleComplex inv(DoubleDoubleComplex i_x) {
        return new DoubleDoubleComplex(realFactory.inv(i_x.re()));
    }

    @Override
    public DoubleDoubleComplex u(DoubleDoubleComplex i_x) {
        return new DoubleDoubleComplex(realFactory.u(i_x.re()));
    }

    @Override
    public DoubleDoubleComplex uramp(DoubleDoubleComplex i_x) {
        return new DoubleDoubleComplex(realFactory.uramp(i_x.re()));
    }

    @Override
    public DoubleDoubleComplex pow(DoubleDoubleComplex value, DoubleDoubleComplex pow) {
        return value.pow(pow);
    }

    @Override
    public DoubleDoubleComplex pwr(DoubleDoubleComplex i_x, DoubleDoubleComplex i_y) {
        return i_x.pwr(i_y);
    }

    @Override
    public DoubleDoubleComplex pwrs(DoubleDoubleComplex i_x, DoubleDoubleComplex i_y) {
        return new DoubleDoubleComplex(realFactory.pwrs(i_x.re(), i_y.re()));
    }

    @Override
    public DoubleDoubleComplex sqrt(DoubleDoubleComplex value) {
        return value.sqrt();
    }

    @Override
    public DoubleDoubleComplex square(DoubleDoubleComplex value) {
        return value.pow(2);
    }

    @Override
    public DoubleDoubleComplex hypot(DoubleDoubleComplex i_x, DoubleDoubleComplex i_y) {
        return i_x.pow(2).plus(i_y.pow(2)).sqrt();
    }

    @Override
    public DoubleDoubleComplex floor(DoubleDoubleComplex value) {
        return new DoubleDoubleComplex(realFactory.floor(value.re()));
    }

    @Override
    public DoubleDoubleComplex ceil(DoubleDoubleComplex value) {
        return new DoubleDoubleComplex(realFactory.ceil(value.re()));
    }

    @Override
    public DoubleDoubleComplex round(DoubleDoubleComplex value) {
        return new DoubleDoubleComplex(realFactory.round(value.re()));
    }
}
