package org.nd4j.autodiff;

public class DoubleReal implements RealNumber<DoubleReal> {

    private double x;

    public DoubleReal() {
        this(0.0);
    }

    public DoubleReal(double x) {
        this.x = x;
    }

    public DoubleReal(String doubleString) {
        x = Double.parseDouble(doubleString);
    }

    public void set(double value) {
        this.x = value;
    }

    public double doubleValue() {
        return x;
    }

    public double modulus() {
        return Math.abs(x);
    }

    @Override
    public String toString() {
        return String.valueOf(x);
    }

    @Override
    public DoubleReal inverse() {
        return new DoubleReal(1.0 / x);
    }

    @Override
    public DoubleReal negate() {
        return new DoubleReal(-x);
    }

    // Operators for DoubleReal

    @Override
    public DoubleReal plus(DoubleReal rd) {
        return new DoubleReal(x + rd.x);
    }

    @Override
    public DoubleReal minus(DoubleReal rd) {
        return new DoubleReal(x - rd.x);
    }

    @Override
    public DoubleReal mul(DoubleReal rd) {
        return new DoubleReal(x * rd.x);
    }

    @Override
    public DoubleReal div(DoubleReal rd) {
        return new DoubleReal(x / rd.x);
    }

    public DoubleReal pow(DoubleReal a) {
        return pow(a.doubleValue());
    }

    public DoubleReal floor() {
        return new DoubleReal(Math.floor(x));
    }

    public DoubleReal ceil() {
        return new DoubleReal(Math.ceil(x));
    }

    public DoubleReal round() {
        return new DoubleReal(Math.round(x));
    }

    public DoubleReal abs() {
        return new DoubleReal(Math.abs(x));
    }

    public DoubleReal sqrt() {
        return new DoubleReal(Math.sqrt(x));
    }
    // Operators for double

    public DoubleReal plus(double v) {
        return new DoubleReal(x + v);
    }

    public DoubleReal minus(double v) {
        return new DoubleReal(x - v);
    }

    public DoubleReal prod(double v) {
        return new DoubleReal(x * v);
    }

    public DoubleReal div(double v) {
        return new DoubleReal(x / v);
    }

    public DoubleReal pow(double v) {
        return new DoubleReal(Math.pow(x, v));
    }

    public DoubleReal cos() {
        return new DoubleReal(Math.cos(x));
    }

    public DoubleReal acos() {
        return new DoubleReal(Math.acos(x));
    }

    public DoubleReal cosh() {
        return new DoubleReal(Math.cosh(x));
    }

    public DoubleReal acosh() {
        return new DoubleReal(Math.log(Math.sqrt(Math.pow(x, 2) - 1) + x));
    }

    public DoubleReal sin() {
        return new DoubleReal(Math.sin(x));
    }

    public DoubleReal asin() {
        return new DoubleReal(Math.asin(x));
    }

    public DoubleReal sinh() {
        return new DoubleReal(Math.sinh(x));
    }

    public DoubleReal asinh() {
        return new DoubleReal(Math.log(Math.sqrt(Math.pow(x, 2) + 1) + x));
    }

    public DoubleReal tan() {
        return new DoubleReal(Math.tan(x));
    }

    public DoubleReal atan() {
        return new DoubleReal(Math.atan(x));
    }

    public DoubleReal tanh() {
        return new DoubleReal(Math.tanh(x));
    }

    public DoubleReal atanh() {
        return new DoubleReal(Math.log((x + 1) / (1 - x)) * 0.5);
    }

    public DoubleReal exp() {
        return new DoubleReal(Math.exp(x));
    }

    public DoubleReal log() {
        return new DoubleReal(Math.log(x));
    }

    public DoubleReal log10() {
        return new DoubleReal(Math.log10(x));
    }

    public DoubleReal sgn() {
        return new DoubleReal(Math.signum(x));
    }

    public DoubleReal pwr(DoubleReal y) {
        return new DoubleReal(Math.pow(Math.abs(x), y.doubleValue()));
    }

    public DoubleReal pwrs(DoubleReal y) {
        return new DoubleReal(Math.pow(Math.abs(x), y.doubleValue()) * Math.signum(x));
    }

    public DoubleReal square() {
        return new DoubleReal(x * x);
    }


    @Override
    public DoubleReal pow(int n) {
        return new DoubleReal(Math.pow(x, n));
    }

    @Override
    public DoubleReal mul(long n) {
        return new DoubleReal(x * n);
    }

    @Override
    public double getReal() {
        return x;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        long temp;
        temp = Double.doubleToLongBits(x);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        DoubleReal other = (DoubleReal) obj;
        if (Double.doubleToLongBits(x) != Double.doubleToLongBits(other.x))
            return false;
        return true;
    }
}
