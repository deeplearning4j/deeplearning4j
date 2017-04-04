package org.nd4j.autodiff;

import org.apache.commons.math3.util.FastMath;

import org.nd4j.autodiff.doubledouble.DoubleDouble;
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;

public class DoubleDoubleComplex implements ComplexNumber<DoubleDoubleReal, DoubleDoubleComplex> {

    public static final DoubleDoubleComplex I = new DoubleDoubleComplex(0.0, 1.0);
    public static final DoubleDoubleComplex ONE = new DoubleDoubleComplex(1.0, 0.0);
    public static final DoubleDoubleComplex ZERO = new DoubleDoubleComplex(0.0, 0.0);

    private static final DoubleDoubleComplex TEN = new DoubleDoubleComplex(10.0, 0.0);
    private static final DoubleDoubleComplex LOG_TEN = TEN.log();
    private static final DoubleDoubleComplex NAN = new DoubleDoubleComplex(
            new DoubleDoubleReal(DoubleDouble.NaN), new DoubleDoubleReal(DoubleDouble.NaN));

    private final DoubleDoubleReal imaginary;
    private final DoubleDoubleReal real;

    public DoubleDoubleComplex() {
        this(0, 0);
    }

    public DoubleDoubleComplex(double real) {
        this(real, 0.0);
    }

    public DoubleDoubleComplex(double real, double imaginary) {
        this(new DoubleDoubleReal(real), new DoubleDoubleReal(imaginary));
    }

    public DoubleDoubleComplex(DoubleDoubleReal real) {
        this(real, new DoubleDoubleReal());
    }

    public DoubleDoubleComplex(DoubleDouble real, DoubleDouble imaginary) {
        this(new DoubleDoubleReal(real), new DoubleDoubleReal(imaginary));
    }

    public DoubleDoubleComplex(DoubleDoubleReal real, DoubleDoubleReal imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }

    @Override
    public double getImaginary() {
        return imaginary.doubleValue();
    }

    @Override
    public double getReal() {
        return real.doubleValue();
    }

    @Override
    public DoubleDoubleComplex inverse() {
        return ONE.div(this);
    }

    @Override
    public DoubleDoubleComplex div(DoubleDoubleComplex divisor) {
        DoubleDoubleReal denominator = divisor.real.pow(2).plus(divisor.imaginary.pow(2));
        DoubleDoubleReal realPart = real.mul(divisor.real)
                                        .plus(imaginary.mul(divisor.imaginary))
                                        .div(denominator);
        DoubleDoubleReal imaginaryPart = imaginary.mul(divisor.real)
                                                  .minus(real.mul(divisor.imaginary))
                                                  .div(denominator);
        return new DoubleDoubleComplex(realPart, imaginaryPart);
    }

    @Override
    public DoubleDoubleComplex plus(DoubleDoubleComplex plusend) {
        return new DoubleDoubleComplex(real.plus(plusend.real), imaginary.plus(plusend.imaginary));
    }

    @Override
    public DoubleDoubleComplex minus(DoubleDoubleComplex subtrahend) {
        return new DoubleDoubleComplex(real.minus(subtrahend.real),
                imaginary.minus(subtrahend.imaginary));
    }

    @Override
    public DoubleDoubleComplex mul(long factor) {
        return mul(new DoubleDoubleComplex(factor));
    }

    @Override
    public DoubleDoubleComplex negate() {
        return new DoubleDoubleComplex(real.negate(), imaginary.negate());
    }

    @Override
    public DoubleDoubleComplex mul(DoubleDoubleComplex factor) {
        return new DoubleDoubleComplex(real.mul(factor.real).minus(imaginary.mul(factor.imaginary)),
                real.mul(factor.imaginary).plus(imaginary.mul(factor.real)));
    }

    public DoubleDoubleComplex abs() {
        return new DoubleDoubleComplex(real.pow(2).plus(imaginary.pow(2)).sqrt(),
                DoubleDoubleReal.ZERO);
    }

    @Override
    public DoubleDoubleComplex pow(int n) {
        if (this.equals(ZERO)) {
            if (n == 0) {
                return NAN;
            } else {
                return ZERO;
            }
        } else if (n == 0) {
            return ONE;
        } else {
            DoubleDoubleComplex result = ZERO;
            for (int k = 0; k <= n; k++) {
                result = result.plus(powI(k).mul(binomialCoeff(n, k))
                                            .mul(new DoubleDoubleComplex(real.pow(n - k),
                                                    DoubleDoubleReal.ZERO))
                                            .mul(new DoubleDoubleComplex(imaginary.pow(k),
                                                    DoubleDoubleReal.ZERO)));
            }
            return result;
        }
    }

    public DoubleDoubleComplex exp() {
        DoubleDoubleReal expReal = real.exp();
        return new DoubleDoubleComplex(expReal.mul(imaginary.cos()), expReal.mul(imaginary.sin()));
    }

    public DoubleDoubleComplex log() {
        return new DoubleDoubleComplex(this.abs().re().log(),
                new DoubleDoubleReal(FastMath.atan2(imaginary.doubleValue(), real.doubleValue())));
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(imaginary, real);
    }

    @Override
    public boolean equals(Object object) {
        if (object instanceof DoubleDoubleComplex) {
            DoubleDoubleComplex that = (DoubleDoubleComplex) object;
            return Objects.equal(this.imaginary, that.imaginary) && Objects.equal(this.real,
                    that.real);
        }
        return false;
    }

    @Override
    public String toString() {
        return MoreObjects.toStringHelper(this)
                          .add("imaginary", imaginary)
                          .add("real", real)
                          .toString();
    }

    @Override
    public DoubleDoubleComplex conjugate() {
        return new DoubleDoubleComplex(real, imaginary.negate());
    }

    @Override
    public DoubleDoubleReal re() {
        return real;
    }

    @Override
    public DoubleDoubleReal im() {
        return imaginary;
    }

    public DoubleDoubleComplex cos() {
        DoubleDoubleComplex d1 = new DoubleDoubleComplex(imaginary.negate(), real).exp();
        DoubleDoubleComplex d2 = new DoubleDoubleComplex(imaginary, real.negate()).exp();
        return d1.plus(d2).mul(new DoubleDoubleComplex(.5));
    }

    public DoubleDoubleComplex sin() {
        DoubleDoubleComplex d1 = new DoubleDoubleComplex(imaginary, real.negate()).exp();
        DoubleDoubleComplex d2 = new DoubleDoubleComplex(imaginary.negate(), real).exp();
        return d1.minus(d2).mul(new DoubleDoubleComplex(0, .5));
    }

    public DoubleDoubleComplex tan() {
        return sin().div(cos());
    }

    public DoubleDoubleComplex pow(DoubleDoubleComplex pow) {
        return log().mul(pow).exp();
    }

    public DoubleDoubleComplex sqrt() {
        DoubleDoubleReal realPart = real.plus(abs().re()).div(DoubleDoubleReal.TWO).sqrt();
        DoubleDoubleReal imaginaryPart = abs().re().minus(real).div(DoubleDoubleReal.TWO).sqrt();
        return new DoubleDoubleComplex(realPart, imaginaryPart);
    }

    public DoubleDoubleComplex acos() {
        return this.plus(ONE.minus(this.pow(2)).sqrt().mul(I)).log().mul(I.negate());
    }

    public DoubleDoubleComplex asin() {
        return ONE.minus(this.pow(2)).sqrt().plus(this.mul(I)).log().mul(I.negate());
    }

    public DoubleDoubleComplex atan() {
        return this.plus(I).div(I.minus(this)).log().mul(I.div(new DoubleDoubleComplex(2.0, 0.0)));
    }

    public DoubleDoubleComplex acosh() {
        return this.plus(this.plus(ONE).sqrt().mul(this.minus(ONE).sqrt())).log();
    }

    public DoubleDoubleComplex asinh() {
        return this.plus(ONE.plus(this.pow(2)).sqrt()).log();
    }

    public DoubleDoubleComplex cosh() {
        return new DoubleDoubleComplex(real.cosh().mul(imaginary.cos()),
                real.sinh().mul(imaginary.sin()));
    }

    public DoubleDoubleComplex sinh() {
        return new DoubleDoubleComplex(real.sinh().mul(imaginary.cos()),
                real.cosh().mul(imaginary.sin()));
    }

    public DoubleDoubleComplex tanh() {
        DoubleDoubleReal real2 = real.mul(2);
        DoubleDoubleReal imaginary2 = imaginary.mul(2);
        DoubleDoubleReal d = real2.cosh().plus(imaginary2.cos());
        return new DoubleDoubleComplex(real2.sinh().div(d), imaginary2.sin().div(d));
    }

    public DoubleDoubleComplex atanh() {
        return ONE.plus(this).log().minus(ONE.minus(this).log()).div(new DoubleDoubleComplex(2));
    }

    public DoubleDoubleComplex log10() {
        return this.log().div(LOG_TEN);
    }

    public DoubleDoubleComplex pwr(DoubleDoubleComplex i_y) {
        return this.abs().pow(i_y);
    }

    private DoubleDoubleComplex binomialCoeff(int n, int k) {
        int res = 1;
        if (k > n - k)
            k = n - k;
        for (int i = 0; i < k; ++i) {
            res *= (n - i);
            res /= (i + 1);
        }
        return new DoubleDoubleComplex(res);
    }

    private DoubleDoubleComplex powI(int n) {
        switch (n % 4) {
        case 0:
            return ONE;
        case 1:
            return I;
        case 2:
            return ONE.negate();
        case 3:
            return I.negate();
        }
        throw null;
    }
}
