package org.nd4j.autodiff;

import org.nd4j.autodiff.doubledouble.DoubleDouble;
import com.google.common.base.Objects;

public class DoubleDoubleReal
        implements RealNumber<DoubleDoubleReal>, Comparable<DoubleDoubleReal> {

    public static final DoubleDoubleReal TWO = new DoubleDoubleReal(DoubleDouble.TWO);
    public static final DoubleDoubleReal ONE = new DoubleDoubleReal(DoubleDouble.ONE);
    public static final DoubleDoubleReal HALF = new DoubleDoubleReal(DoubleDouble.HALF);
    public static final DoubleDoubleReal ZERO = new DoubleDoubleReal(DoubleDouble.ZERO);

    private DoubleDouble doubleDouble;

    public DoubleDoubleReal() {
        doubleDouble = DoubleDouble.ZERO;
    }

    public DoubleDoubleReal(double value) {
        doubleDouble = DoubleDouble.fromOneDouble(value);
    }

    public DoubleDoubleReal(String doubleString) {
        doubleDouble = DoubleDouble.fromString(doubleString);
    }

    public DoubleDoubleReal(DoubleDouble bigDecimal) {
        this.doubleDouble = bigDecimal;
    }

    public void set(double value) {
        doubleDouble = DoubleDouble.fromOneDouble(value);
    }

    public double doubleValue() {
        return doubleDouble.doubleValue();
    }

    public DoubleDouble getDoubleDouble() {
        return doubleDouble;
    }

    public DoubleDouble modulus() {
        return doubleDouble.abs();
    }

    @Override
    public String toString() {
        return doubleDouble.toString();
    }

    @Override
    public DoubleDoubleReal inverse() {
        return new DoubleDoubleReal(DoubleDouble.ONE.divide(doubleDouble));
    }

    @Override
    public DoubleDoubleReal negate() {
        return new DoubleDoubleReal(doubleDouble.negate());
    }

    public DoubleDoubleReal pow(DoubleDoubleReal value) {
        return new DoubleDoubleReal(doubleDouble.pow(value.getDoubleDouble()));
    }

    // Operators for DoubleReal
    @Override
    public DoubleDoubleReal plus(DoubleDoubleReal value) {
        return new DoubleDoubleReal(doubleDouble.add(value.doubleDouble));
    }

    @Override
    public DoubleDoubleReal minus(DoubleDoubleReal value) {
        return new DoubleDoubleReal(doubleDouble.subtract(value.doubleDouble));
    }

    @Override
    public DoubleDoubleReal mul(DoubleDoubleReal value) {
        return new DoubleDoubleReal(doubleDouble.multiply(value.doubleDouble));
    }

    @Override
    public DoubleDoubleReal div(DoubleDoubleReal value) {
        return new DoubleDoubleReal(doubleDouble.divide(value.doubleDouble));
    }

    @Override
    public DoubleDoubleReal pow(int value) {
        return new DoubleDoubleReal(doubleDouble.pow(value));
    }

    @Override
    public DoubleDoubleReal mul(long value) {
        return new DoubleDoubleReal(doubleDouble.multiply(DoubleDouble.fromOneDouble(value)));
    }

    @Override
    public double getReal() {
        return doubleDouble.doubleValue();
    }

    public DoubleDoubleReal floor() {
        return new DoubleDoubleReal(doubleDouble.floor());
    }

    public DoubleDoubleReal ceil() {
        return new DoubleDoubleReal(doubleDouble.ceil());
    }

    public DoubleDoubleReal abs() {
        return new DoubleDoubleReal(doubleDouble.abs());
    }

    public DoubleDoubleReal round() {
        return new DoubleDoubleReal(doubleDouble.round());
    }

    public DoubleDoubleReal sqrt() {
        return new DoubleDoubleReal(doubleDouble.sqrt());
    }

    public DoubleDoubleReal log() {
        return new DoubleDoubleReal(doubleDouble.log());
    }

    public DoubleDoubleReal acosh() {
        return this.pow(2).minus(DoubleDoubleReal.ONE).sqrt().plus(this).log();
    }

    public DoubleDoubleReal sin() {
        return new DoubleDoubleReal(doubleDouble.sin());
    }

    public DoubleDoubleReal cos() {
        return new DoubleDoubleReal(doubleDouble.cos());
    }

    public DoubleDoubleReal acos() {
        return new DoubleDoubleReal(doubleDouble.acos());
    }

    public DoubleDoubleReal cosh() {
        return new DoubleDoubleReal(doubleDouble.cosh());
    }

    public DoubleDoubleReal asin() {
        return new DoubleDoubleReal(doubleDouble.asin());
    }

    public DoubleDoubleReal sinh() {
        return new DoubleDoubleReal(doubleDouble.sinh());
    }

    public DoubleDoubleReal asinh() {
        return new DoubleDoubleReal(this.getDoubleDouble()
                                        .pow(2)
                                        .add(DoubleDouble.ONE)
                                        .sqrt()
                                        .add(this.getDoubleDouble())
                                        .log());
    }

    public DoubleDoubleReal tan() {
        return new DoubleDoubleReal(doubleDouble.tan());
    }

    public DoubleDoubleReal atan() {
        return new DoubleDoubleReal(doubleDouble.atan());
    }

    public DoubleDoubleReal tanh() {
        return this.sinh().div(this.cosh());
    }

    public DoubleDoubleReal atanh() {
        return this.plus(ONE).div(ONE.minus(this)).log().div(TWO);
    }

    public DoubleDoubleReal exp() {
        return new DoubleDoubleReal(this.getDoubleDouble().exp());
    }

    public DoubleDoubleReal log10() {
        return new DoubleDoubleReal(doubleDouble.log10());
    }

    public DoubleDoubleReal sgn() {
        return new DoubleDoubleReal(this.getDoubleDouble().signum());
    }

    public DoubleDoubleReal pwr(DoubleDoubleReal n) {
        return this.abs().pow(n);
    }

    public DoubleDoubleReal pwrs(DoubleDoubleReal n) {
        return this.abs().pow(n).mul(this.sgn());
    }

    public DoubleDoubleReal square() {
        return new DoubleDoubleReal(this.getDoubleDouble().sqr());
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(doubleDouble);
    }

    @Override
    public boolean equals(Object object) {
        if (object instanceof DoubleDoubleReal) {
            DoubleDoubleReal that = (DoubleDoubleReal) object;
            return Objects.equal(this.doubleDouble, that.doubleDouble);
        }
        return false;
    }

    @Override
    public int compareTo(DoubleDoubleReal other) {
        return doubleDouble.compareTo(other.doubleDouble);
    }
}
