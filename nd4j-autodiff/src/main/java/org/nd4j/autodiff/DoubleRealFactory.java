package org.nd4j.autodiff;

import java.util.Random;

public class DoubleRealFactory implements AbstractFactory<DoubleReal> {

    private static final DoubleRealFactory m_INSTANCE = new DoubleRealFactory();
    private Random randomGenerator = new Random();

    private DoubleRealFactory() {
    }


    public static DoubleRealFactory instance() {
        return m_INSTANCE;
    }


    @Override
    public DoubleReal val(double v) {
        return new DoubleReal(v);
    }

    @Override
    public DoubleReal abs(DoubleReal x) {
        return x.abs();
    }

    @Override
    public DoubleReal min(DoubleReal x, DoubleReal y) {
        return x.doubleValue() < y.doubleValue() ? new DoubleReal(
                x.doubleValue()) : new DoubleReal(y.doubleValue());
    }

    @Override
    public DoubleReal max(DoubleReal x, DoubleReal y) {
        return x.doubleValue() > y.doubleValue() ? new DoubleReal(
                x.doubleValue()) : new DoubleReal(y.doubleValue());
    }

    private static final DoubleReal m_ZERO = new DoubleReal(0.0);
    private static final DoubleReal m_UNIT = new DoubleReal(1.0);

    @Override
    public DoubleReal zero() {
        return m_ZERO;
    }

    @Override
    public DoubleReal one() {
        return m_UNIT;
    }

    @Override
    public DoubleReal cos(DoubleReal x) {
        return x.cos();
    }

    @Override
    public DoubleReal acos(DoubleReal x) {
        return x.acos();
    }

    @Override
    public DoubleReal cosh(DoubleReal x) {
        return x.cosh();
    }

    @Override
    public DoubleReal acosh(DoubleReal x) {
        return x.acosh();
    }

    @Override
    public DoubleReal sin(DoubleReal x) {
        return x.sin();
    }

    @Override
    public DoubleReal asin(DoubleReal x) {
        return x.asin();
    }

    @Override
    public DoubleReal sinh(DoubleReal x) {
        return x.sinh();
    }

    @Override
    public DoubleReal asinh(DoubleReal x) {
        return x.asinh();
    }

    @Override
    public DoubleReal tan(DoubleReal x) {
        return x.tan();
    }

    @Override
    public DoubleReal atan(DoubleReal x) {
        return x.atan();
    }

    @Override
    public DoubleReal atan2(DoubleReal x, DoubleReal y) {
        return new DoubleReal(Math.atan2(x.doubleValue(), y.doubleValue()));
    }

    @Override
    public DoubleReal tanh(DoubleReal x) {
        return x.tanh();
    }

    @Override
    public DoubleReal atanh(DoubleReal x) {
        return x.atanh();
    }

    @Override
    public DoubleReal exp(DoubleReal x) {
        return x.exp();
    }

    @Override
    public DoubleReal log(DoubleReal x) {
        return x.log();
    }

    @Override
    public DoubleReal log10(DoubleReal x) {
        return x.log10();
    }

    @Override
    public DoubleReal flat(DoubleReal x) {
        double xValue = x.doubleValue();
        return new DoubleReal(-xValue + (xValue + xValue) * randomGenerator.nextDouble());
    }

    @Override
    public DoubleReal mc(DoubleReal x, DoubleReal y) {
        double max = Math.max(x.doubleValue() * (1 + y.doubleValue()),
                x.doubleValue() * (1 - y.doubleValue()));
        double min = Math.min(x.doubleValue() * (1 + y.doubleValue()),
                x.doubleValue() * (1 - y.doubleValue()));
        return new DoubleReal(min + (max - min) * randomGenerator.nextDouble());
    }

    @Override
    public DoubleReal rand(DoubleReal x) {
        return new DoubleReal(new Random((long) x.getReal()).nextDouble());
    }

    @Override
    public DoubleReal random(DoubleReal x) {
        return new DoubleReal(new Random((long) x.getReal()).nextDouble());
    }

    @Override
    public DoubleReal gauss(DoubleReal x) {
        return new DoubleReal(randomGenerator.nextGaussian()*x.doubleValue());
    }

    @Override
    public DoubleReal sgn(DoubleReal x) {
        return x.sgn();
    }

    @Override
    public DoubleReal ifx(DoubleReal x, DoubleReal y, DoubleReal z) {
        return x.doubleValue() > .5 ? y : z;
    }

    @Override
    public DoubleReal buf(DoubleReal x) {
        return x.doubleValue() > .5 ? new DoubleReal(1) : new DoubleReal(0);
    }

    @Override
    public DoubleReal inv(DoubleReal x) {
        return x.doubleValue() > .5 ? new DoubleReal(0) : new DoubleReal(1);
    }

    @Override
    public DoubleReal u(DoubleReal x) {
        return x.doubleValue() > 0 ? new DoubleReal(1) : new DoubleReal(0);
    }

    @Override
    public DoubleReal uramp(DoubleReal x) {
        return x.doubleValue() > 0 ? new DoubleReal(x.doubleValue()) : new DoubleReal(0);
    }

    @Override
    public DoubleReal pow(DoubleReal x, DoubleReal y) {
        return x.pow(y);
    }

    @Override
    public DoubleReal pwr(DoubleReal x, DoubleReal y) {
        return x.pwr(y);
    }

    @Override
    public DoubleReal pwrs(DoubleReal x, DoubleReal y) {
        return x.pwrs(y);
    }

    @Override
    public DoubleReal sqrt(DoubleReal x) {
        return x.sqrt();
    }

    @Override
    public DoubleReal square(DoubleReal x) {
        return x.square();
    }

    @Override
    public DoubleReal hypot(DoubleReal x, DoubleReal y) {
        return x.pow(2).plus(y.pow(2)).sqrt();
    }

    @Override
    public DoubleReal floor(DoubleReal value) {
        return value.floor();
    }

    @Override
    public DoubleReal ceil(DoubleReal value) {
        return value.ceil();
    }

    @Override
    public DoubleReal round(DoubleReal value) {
        return value.round();
    }

}
