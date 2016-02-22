/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.complex;

import org.nd4j.linalg.factory.Nd4j;

/**
 * Base class for complex doubles
 *
 * @author Adam Gibson
 */
public abstract class BaseComplexDouble implements IComplexDouble {
    protected double real, imag;

    public BaseComplexDouble() {
    }

    public BaseComplexDouble(Double real, Double imag) {
        this.real = real;
        this.imag = imag;
    }

    public BaseComplexDouble(double real, double imag) {
        this.real = real;
        this.imag = imag;
    }

    public BaseComplexDouble(double real) {
        this(real, 0);
    }

    @Override
    public IComplexNumber dup() {
        return Nd4j.createComplexNumber(real, imag);
    }


    @Override
    public IComplexNumber eqc(IComplexNumber num) {
        double val = num.realComponent().doubleValue();
        double imag = num.imaginaryComponent().doubleValue();
        double otherVal = num.realComponent().doubleValue();
        double otherImag = num.imaginaryComponent().doubleValue();
        if (val == otherVal)
            return Nd4j.createComplexNumber(1, 0);
        else if (val != otherVal)
            return Nd4j.createComplexNumber(0, 0);
        else if (imag == otherImag)
            return Nd4j.createComplexNumber(1, 0);
        else
            return Nd4j.createComplexNumber(0, 0);
    }

    @Override
    public IComplexNumber neqc(IComplexNumber num) {
        double val = num.realComponent().doubleValue();
        double imag = num.imaginaryComponent().doubleValue();
        double otherVal = num.realComponent().doubleValue();
        double otherImag = num.imaginaryComponent().doubleValue();
        if (val != otherVal)
            return Nd4j.createComplexNumber(1, 0);
        else if (val == otherVal)
            return Nd4j.createComplexNumber(0, 0);
        else if (imag != otherImag)
            return Nd4j.createComplexNumber(1, 0);
        else
            return Nd4j.createComplexNumber(0, 0);
    }

    @Override
    public IComplexNumber gt(IComplexNumber num) {
        double val = num.realComponent().doubleValue();
        double imag = num.imaginaryComponent().doubleValue();
        double otherVal = num.realComponent().doubleValue();
        double otherImag = num.imaginaryComponent().doubleValue();
        if (val > otherVal)
            return Nd4j.createComplexNumber(1, 0);
        else if (val < otherVal)
            return Nd4j.createComplexNumber(0, 0);
        else if (imag > otherImag)
            return Nd4j.createComplexNumber(1, 0);
        else
            return Nd4j.createComplexNumber(0, 0);
    }

    @Override
    public IComplexNumber lt(IComplexNumber num) {
        double val = num.realComponent().doubleValue();
        double imag = num.imaginaryComponent().doubleValue();
        double otherVal = num.realComponent().doubleValue();
        double otherImag = num.imaginaryComponent().doubleValue();
        if (val < otherVal)
            return Nd4j.createComplexNumber(1, 0);
        else if (val > otherVal)
            return Nd4j.createComplexNumber(0, 0);
        else if (imag < otherImag)
            return Nd4j.createComplexNumber(1, 0);
        else
            return Nd4j.createComplexNumber(0, 0);
    }

    /**
     * Convert to a double
     *
     * @return this complex number as a double
     */
    @Override
    public IComplexDouble asDouble() {
        return this;
    }


    @Override
    public IComplexDouble conji() {
        set(realComponent(), -imaginaryComponent());
        return this;
    }

    @Override
    public IComplexNumber conj() {
        return dup().conji();
    }

    @Override
    public IComplexNumber set(Number real, Number imag) {
        this.real = real.doubleValue();
        this.imag = imag.doubleValue();
        return this;
    }

    @Override
    public IComplexNumber copy(IComplexNumber other) {
        return Nd4j.createDouble(other.realComponent().doubleValue(), other.imaginaryComponent().doubleValue());

    }

    @Override
    public IComplexNumber set(IComplexNumber set) {
        return set(set.realComponent().doubleValue(), set.imaginaryComponent().doubleValue());
    }

    @Override
    public IComplexNumber rsubi(IComplexNumber c) {
        return rsubi(c, this);
    }

    @Override
    public IComplexNumber rsub(IComplexNumber c) {
        return dup().rsubi(c);
    }

    @Override
    public IComplexNumber rsubi(IComplexNumber a, IComplexNumber result) {
        return result.set(a.sub(this));
    }

    @Override
    public IComplexNumber rsubi(Number a, IComplexNumber result) {
        return result.set(a.doubleValue() - result.realComponent().doubleValue(), imaginaryComponent());
    }

    @Override
    public IComplexNumber rsubi(Number a) {
        return set(a.doubleValue() - realComponent().doubleValue(), imaginaryComponent());
    }

    @Override
    public IComplexNumber rsub(Number r) {
        return dup().rsubi(r);
    }

    @Override
    public IComplexNumber rdiv(IComplexNumber c) {
        return dup().rdivi(c);
    }

    @Override
    public IComplexNumber rdivi(IComplexNumber c, IComplexNumber result) {
        return result.set(c.realComponent().doubleValue() / realComponent().doubleValue(), c.imaginaryComponent().doubleValue() / imaginaryComponent().doubleValue());
    }

    @Override
    public IComplexNumber rdivi(IComplexNumber c) {
        return rdivi(c, this);
    }

    @Override
    public IComplexNumber rdivi(Number v, IComplexNumber result) {
        double d = realComponent().doubleValue() * realComponent().doubleValue() + imaginaryComponent().doubleValue() * imaginaryComponent().doubleValue();
        return result.set(v.doubleValue() * result.realComponent().doubleValue() / d, -v.doubleValue() * result.imaginaryComponent().doubleValue() / d);
    }

    @Override
    public IComplexNumber rdivi(Number v) {
        return set(v.doubleValue() / realComponent().doubleValue(), imaginaryComponent());
    }

    @Override
    public IComplexNumber rdiv(Number v) {
        return dup().rdivi(v);
    }

    @Override
    public IComplexFloat asFloat() {
        return Nd4j.createFloat(realComponent().floatValue(), imaginaryComponent().floatValue());
    }

    /**
     * Add two complex numbers in-place
     *
     * @param c
     * @param result
     */
    @Override
    public IComplexNumber addi(IComplexNumber c, IComplexNumber result) {
        result.set(realComponent().doubleValue() + c.realComponent().doubleValue(),
                    imaginaryComponent().doubleValue() + c.imaginaryComponent().doubleValue());
        return this;
    }

    /**
     * Add two complex numbers in-place storing the result in this.
     *
     * @param c
     */
    @Override
    public IComplexNumber addi(IComplexNumber c) {
        return addi(c, this);
    }

    /**
     * Add two complex numbers.
     *
     * @param c
     */
    @Override
    public IComplexNumber add(IComplexNumber c) {
        return dup().addi(c);
    }

    /**
     * Add a realComponent number to a complex number in-place.
     *
     * @param a
     * @param result
     */
    @Override
    public IComplexNumber addi(Number a, IComplexNumber result) {
        return result.set(result.realComponent().doubleValue() + a.doubleValue(), result.imaginaryComponent().doubleValue());
    }

    /**
     * Add a realComponent number to complex number in-place, storing the result in this.
     *
     * @param c
     */
    @Override
    public IComplexNumber addi(Number c) {
        return addi(c, this);
    }

    /**
     * Add a realComponent number to a complex number.
     *
     * @param c
     */
    @Override
    public IComplexNumber add(Number c) {
        return dup().addi(c);
    }

    /**
     * Subtract two complex numbers, in-place
     *
     * @param c
     * @param result
     */
    @Override
    public IComplexNumber subi(IComplexNumber c, IComplexNumber result) {
        return result.set(realComponent().doubleValue() - c.realComponent().doubleValue(), imaginaryComponent().doubleValue() - c.imaginaryComponent().doubleValue());
    }

    @Override
    public IComplexNumber subi(IComplexNumber c) {
        return subi(c, this);
    }

    /**
     * Subtract two complex numbers
     *
     * @param c
     */
    @Override
    public IComplexNumber sub(IComplexNumber c) {
        return dup().subi(c);
    }

    @Override
    public IComplexNumber subi(Number a, IComplexNumber result) {
        return result.set(realComponent().doubleValue() - a.doubleValue(), imaginaryComponent().doubleValue());
    }

    @Override
    public IComplexNumber subi(Number a) {
        return subi(a, this);
    }

    @Override
    public IComplexNumber sub(Number r) {
        return dup().subi(r);
    }

    /**
     * Multiply two complex numbers, inplace
     *
     * @param c
     * @param result
     */
    @Override
    public IComplexNumber muli(IComplexNumber c, IComplexNumber result) {
        double newR = real * c.realComponent().doubleValue() - imag * c.imaginaryComponent().doubleValue();
        double newI = real * c.imaginaryComponent().doubleValue() + imag * c.realComponent().doubleValue();
        result.set(newR, newI);
        return result;
    }

    @Override
    public IComplexNumber muli(IComplexNumber c) {
        return muli(c, this);
    }

    /**
     * Multiply two complex numbers
     *
     * @param c
     */
    @Override
    public IComplexNumber mul(IComplexNumber c) {
        return dup().muli(c);
    }

    @Override
    public IComplexNumber mul(Number v) {
        return dup().muli(v);
    }

    @Override
    public IComplexNumber muli(Number v, IComplexNumber result) {
        return result.set(realComponent().doubleValue() * v.doubleValue(), imaginaryComponent().doubleValue() * v.doubleValue());
    }

    @Override
    public IComplexNumber muli(Number v) {
        return muli(v, this);
    }

    @Override
    public IComplexNumber exp() {
        IComplexNumber result = dup();
        double realExp = Math.exp(realComponent());
        return result.set(realExp * Math.cos(imaginaryComponent()), realExp * Math.sin(imaginaryComponent()));
    }

    @Override
    public IComplexNumber powi(IComplexNumber c, IComplexNumber result) {
        IComplexNumber eval = log().muli(c).exp();
        result.set(eval.realComponent(), eval.imaginaryComponent());
        return result;
    }

    @Override
    public IComplexNumber pow(Number v) { return dup().powi(v); }

    @Override
    public IComplexNumber pow(IComplexNumber c) { return dup().powi(c); }

    @Override
    public IComplexNumber powi(IComplexNumber c) { return dup().powi(c, this); }

    @Override
    public IComplexNumber powi(Number v) { return dup().powi(v, this); }

    @Override
    public IComplexNumber powi(Number v, IComplexNumber result) {
        IComplexNumber eval = log().muli(v).exp();
        result.set(eval.realComponent(), eval.imaginaryComponent());
        return result;
    }

    /**
     * Divide two complex numbers
     *
     * @param c
     */
    @Override
    public IComplexNumber div(IComplexNumber c) {
        return dup().divi(c);
    }

    /**
     * Divide two complex numbers, in-place
     *
     * @param c
     * @param result
     */
    @Override
    public IComplexNumber divi(IComplexNumber c, IComplexNumber result) {
        double d = c.realComponent().doubleValue() * c.realComponent().doubleValue() + c.imaginaryComponent().doubleValue() * c.imaginaryComponent().doubleValue();
        double newR = (realComponent() * c.realComponent().doubleValue() + imaginaryComponent() * c.imaginaryComponent().doubleValue()) / d;
        double newI = (imaginaryComponent() * c.realComponent().doubleValue() - realComponent() * c.imaginaryComponent().doubleValue()) / d;
        result.set(newR, newI);
        return result;
    }

    @Override
    public IComplexNumber divi(IComplexNumber c) {
        return divi(c, this);
    }

    @Override
    public IComplexNumber divi(Number v, IComplexNumber result) {
        return result.set(result.realComponent().doubleValue() / v.doubleValue(), result.imaginaryComponent().doubleValue() / v.doubleValue());
    }

    @Override
    public IComplexNumber divi(Number v) {
        return divi(v, this);
    }

    @Override
    public IComplexNumber div(Number v) {
        return dup().divi(v);
    }

    @Override
    public boolean eq(IComplexNumber c) {
        return realComponent().equals(c.realComponent()) && imaginaryComponent().equals(c.imaginaryComponent());
    }

    @Override
    public boolean ne(IComplexNumber c) {
        return !eq(c);
    }

    @Override
    public boolean isZero() {
        return real == 0;
    }

    @Override
    public boolean isReal() {
        return imag == 0;
    }

    @Override
    public boolean isImag() {
        return real == 0;
    }


    @Override
    public Double realComponent() {
        return real;
    }

    @Override
    public Double imaginaryComponent() {
        return imag;
    }


    @Override
    public IComplexDouble divi(double v) {
        this.imag = imag / v;
        this.real /= v;
        return this;
    }

    @Override
    public IComplexNumber div(double v) {
        return dup().divi(v);
    }

    /**
     * Return the absolute value
     */
    @Override
    public Double absoluteValue() {
        return Math.sqrt(real * real + imag * imag);
    }

    /**
     * Returns the argument of a complex number.
     */
    @Override
    public Double complexArgument() {
        return Math.acos(realComponent() / absoluteValue());
    }

    @Override
    public IComplexDouble invi() {
        double d = realComponent() * realComponent() + imaginaryComponent() * imaginaryComponent();
        set(realComponent() / d, -imaginaryComponent() / d);
        return this;
    }

    @Override
    public IComplexNumber log() {
        IComplexNumber result = dup();
        double real = (double) result.realComponent();
        double imaginary = (double) result.imaginaryComponent();
        double modulus = Math.sqrt(real*real + imaginary*imaginary);
        double arg = Math.atan2(imaginary,real);
        return result.set(Math.log(modulus), arg);
    }

    @Override
    public IComplexNumber inv() {
        return dup().invi();
    }

    @Override
    public IComplexNumber neg() {
        return dup().negi();
    }

    @Override
    public IComplexDouble negi() {
        set(-realComponent(), -imaginaryComponent());
        return this;
    }

    @Override
    public IComplexDouble sqrt() {
        double a = absoluteValue();
        double s2 = Math.sqrt(2);
        double p = Math.sqrt(a + realComponent()) / s2;
        double q = Math.sqrt(a - realComponent()) / s2 * Math.signum(imaginaryComponent());
        return Nd4j.createDouble(p, q);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof IComplexNumber)) return false;

        IComplexNumber that = (IComplexNumber) o;

        if (Math.abs(that.realComponent().doubleValue() - real) > Nd4j.EPS_THRESHOLD)
            return false;
        if(Math.abs(that.imaginaryComponent().doubleValue() - imag) > Nd4j.EPS_THRESHOLD)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        temp = Double.doubleToLongBits(real);
        result = (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(imag);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public String toString() {
        if (imag >= 0) {
            return real + " + " + imag + "i";
        } else {
            return real + " - " + (-imag) + "i";
        }
    }

}
