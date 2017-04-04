package org.nd4j.autodiff;

import java.util.Random;

import org.nd4j.autodiff.doubledouble.DoubleDouble;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class DoubleDoubleRealFactory implements AbstractFactory<DoubleDoubleReal> {

    private static final DoubleDoubleRealFactory m_INSTANCE = new DoubleDoubleRealFactory();

    private Random randomGenerator = new Random();

    private DoubleDoubleRealFactory() {
    }

    public static DoubleDoubleRealFactory instance() {
        return m_INSTANCE;
    }

    @Override
    public DoubleDoubleReal val(double value) {
        return new DoubleDoubleReal(value);
    }

    @Override
    public DoubleDoubleReal abs(DoubleDoubleReal x) {
        return x.abs();
    }

    @Override
    public DoubleDoubleReal min(DoubleDoubleReal x, DoubleDoubleReal y) {
        return x.compareTo(y) < 0 ? x : y;
    }

    @Override
    public DoubleDoubleReal max(DoubleDoubleReal x, DoubleDoubleReal y) {
        return x.compareTo(y) > 0 ? x : y;
    }

    @Override
    public DoubleDoubleReal zero() {
        return DoubleDoubleReal.ZERO;
    }

    @Override
    public DoubleDoubleReal one() {
        return DoubleDoubleReal.ONE;
    }

    @Override
    public DoubleDoubleReal cos(DoubleDoubleReal value) {
        return value.cos();
    }

    @Override
    public DoubleDoubleReal acos(DoubleDoubleReal value) {
        return value.acos();
    }

    @Override
    public DoubleDoubleReal cosh(DoubleDoubleReal value) {
        return value.cosh();
    }

    @Override
    public DoubleDoubleReal acosh(DoubleDoubleReal value) {
        return value.acosh();
    }

    @Override
    public DoubleDoubleReal sin(DoubleDoubleReal value) {
        return value.sin();
    }

    @Override
    public DoubleDoubleReal asin(DoubleDoubleReal value) {
        return value.asin();
    }

    @Override
    public DoubleDoubleReal sinh(DoubleDoubleReal value) {
        return value.sinh();
    }

    @Override
    public DoubleDoubleReal asinh(DoubleDoubleReal value) {
        return value.asinh();
    }

    @Override
    public DoubleDoubleReal tan(DoubleDoubleReal value) {
        return value.tan();
    }

    @Override
    public DoubleDoubleReal atan(DoubleDoubleReal value) {
        return value.atan();
    }

    @Override
    public DoubleDoubleReal atan2(DoubleDoubleReal x, DoubleDoubleReal y) {
        throw new NotImplementedException();
    }

    @Override
    public DoubleDoubleReal tanh(DoubleDoubleReal value) {
        return value.tanh();
    }

    @Override
    public DoubleDoubleReal atanh(DoubleDoubleReal value) {
        return value.atanh();
    }

    @Override
    public DoubleDoubleReal exp(DoubleDoubleReal value) {
        return value.exp();
    }

    @Override
    public DoubleDoubleReal log(DoubleDoubleReal value) {
        return value.log();
    }

    @Override
    public DoubleDoubleReal log10(DoubleDoubleReal value) {
        return value.log10();
    }

    @Override
    public DoubleDoubleReal flat(DoubleDoubleReal x) {
        double xValue = x.doubleValue();
        return new DoubleDoubleReal(DoubleDouble.fromOneDouble(-xValue + (xValue + xValue) * randomGenerator.nextDouble()));
    }

    @Override
    public DoubleDoubleReal mc(DoubleDoubleReal x, DoubleDoubleReal y) {
        double min = Math.min(x.doubleValue() * (1 - y.doubleValue()),
                x.doubleValue() * (1 + y.doubleValue()));
        double max = Math.max(x.doubleValue() * (1 - y.doubleValue()),
                x.doubleValue() * (1 + y.doubleValue()));
        return new DoubleDoubleReal(
                DoubleDouble.fromOneDouble(min + (max + min) * randomGenerator.nextDouble()));
    }

    @Override
    public DoubleDoubleReal rand(DoubleDoubleReal x) {
        return new DoubleDoubleReal(new Random((long) x.getReal()).nextDouble());
    }

    @Override
    public DoubleDoubleReal random(DoubleDoubleReal x) {
        return new DoubleDoubleReal(new Random((long) x.getReal()).nextDouble());
    }

    @Override
    public DoubleDoubleReal gauss(DoubleDoubleReal x) {
        return new DoubleDoubleReal(randomGenerator.nextGaussian()*x.doubleValue());
    }

    @Override
    public DoubleDoubleReal sgn(DoubleDoubleReal value) {
        return value.sgn();
    }

    @Override
    public DoubleDoubleReal ifx(DoubleDoubleReal x, DoubleDoubleReal y, DoubleDoubleReal z) {
        return x.compareTo(DoubleDoubleReal.HALF) > 0 ? y : z;
    }

    @Override
    public DoubleDoubleReal buf(DoubleDoubleReal x) {
        return x.compareTo(DoubleDoubleReal.HALF) > 0 ? one() : zero();
    }

    @Override
    public DoubleDoubleReal inv(DoubleDoubleReal x) {
        return x.compareTo(DoubleDoubleReal.HALF) > 0 ? zero() : one();
    }

    @Override
    public DoubleDoubleReal u(DoubleDoubleReal x) {
        return x.getDoubleDouble().compareTo(DoubleDouble.ZERO) > 0 ? one() : zero();
    }

    @Override
    public DoubleDoubleReal uramp(DoubleDoubleReal x) {
        return x.getDoubleDouble().compareTo(DoubleDouble.ZERO) > 0 ? new DoubleDoubleReal(
                DoubleDouble.fromDoubleDouble(x.getDoubleDouble())) : zero();
    }

    @Override
    public DoubleDoubleReal pow(DoubleDoubleReal value, DoubleDoubleReal n) {
        return value.pow(n);
    }

    @Override
    public DoubleDoubleReal pwr(DoubleDoubleReal value, DoubleDoubleReal n) {
        return value.pwr(n);
    }

    @Override
    public DoubleDoubleReal pwrs(DoubleDoubleReal value, DoubleDoubleReal n) {
        return value.pwrs(n);
    }

    @Override
    public DoubleDoubleReal sqrt(DoubleDoubleReal value) {
        return value.sqrt();
    }

    @Override
    public DoubleDoubleReal square(DoubleDoubleReal value) {
        return value.square();
    }

    @Override
    public DoubleDoubleReal hypot(DoubleDoubleReal x, DoubleDoubleReal y) {
        return x.pow(2).plus(y.pow(2)).sqrt();
    }

    @Override
    public DoubleDoubleReal floor(DoubleDoubleReal value) {
        return value.floor();
    }

    @Override
    public DoubleDoubleReal ceil(DoubleDoubleReal value) {
        return value.ceil();
    }

    @Override
    public DoubleDoubleReal round(DoubleDoubleReal value) {
        return value.round();
    }

}
