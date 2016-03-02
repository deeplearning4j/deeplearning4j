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

package org.nd4j.linalg.cpu.complex;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Double implementation of a complex number.
 * Based on the jblas api by mikio braun
 *
 * @author Adam Gibson
 */
public class ComplexDouble extends org.jblas.ComplexDouble implements IComplexDouble {

    public final static ComplexDouble UNIT = new ComplexDouble(1, 0);
    public final static ComplexDouble NEG = new ComplexDouble(-1, 0);
    public final static ComplexDouble ZERO = new ComplexDouble(0, 0);


    public ComplexDouble(org.jblas.ComplexDouble c) {
        super(c.real(), c.imag());
    }

    public ComplexDouble(double real, double imag) {
        super(real, imag);
    }

    public ComplexDouble(double real) {
        super(real);
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

    @Override
    public IComplexNumber rsubi(IComplexNumber c) {
        return rsubi(c, this);
    }

    @Override
    public IComplexNumber set(IComplexNumber set) {
        return (IComplexNumber) set(set.realComponent().doubleValue(), set.imaginaryComponent().doubleValue());
    }

    @Override
    public IComplexNumber rsubi(IComplexNumber a, IComplexNumber result) {
        return result.set(a.sub(this));
    }

    /**
     * Returns the argument of a complex number.
     */
    @Override
    public double arg() {
        return super.arg();
    }

    /**
     * Return the absolute value
     */
    @Override
    public double abs() {
        return super.abs();
    }

    /**
     * Convert to a float
     *
     * @return this complex number as a float
     */
    @Override
    public IComplexFloat asFloat() {
        return Nd4j.createFloat((float) real(), (float) imag());
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
    public ComplexDouble dup() {
        return new ComplexDouble(realComponent(), imaginaryComponent());
    }


    @Override
    public ComplexDouble conji() {
        super.set(realComponent(), -imaginaryComponent());
        return this;
    }

    @Override
    public ComplexDouble conj() {
        return dup().conji();
    }

    @Override
    public IComplexNumber set(Number real, Number imag) {
        super.set(real.doubleValue(), imag.doubleValue());
        return this;
    }

    @Override
    public IComplexNumber copy(IComplexNumber other) {
        return other.set(this);

    }

    /**
     * Add two complex numbers in-place
     *
     * @param c
     * @param result
     */
    @Override
    public IComplexNumber addi(IComplexNumber c, IComplexNumber result) {
        return result.set(result.realComponent().doubleValue() + c.realComponent().doubleValue(), result.imaginaryComponent().doubleValue() + c.imaginaryComponent().doubleValue());
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
        return result.set(result.realComponent().doubleValue() - c.realComponent().doubleValue(), result.imaginaryComponent().doubleValue() - c.imaginaryComponent().doubleValue());
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
        return result.set(result.realComponent().doubleValue() - a.doubleValue(), result.imaginaryComponent().doubleValue() - a.doubleValue());
    }

    @Override
    public IComplexNumber subi(Number a) {
        return subi(a, this);
    }

    @Override
    public IComplexNumber sub(Number r) {
        return dup().subi(r);
    }

    @Override
    public IComplexNumber rsub(IComplexNumber c) {
        return dup().rsubi(c);
    }

    @Override
    public IComplexNumber rsubi(Number a, IComplexNumber result) {
        return result.set(a.doubleValue() - result.realComponent().doubleValue(), imaginaryComponent());
    }

    @Override
    public IComplexNumber rsubi(Number a) {
        return rsubi(a, this);
    }

    @Override
    public IComplexNumber rsub(Number r) {
        return dup().rsubi(r);
    }

    /**
     * Multiply two complex numbers, inplace
     *
     * @param c
     * @param result
     */
    @Override
    public IComplexNumber muli(IComplexNumber c, IComplexNumber result) {
        double newR = (real() * c.realComponent().doubleValue() - imag() * c.imaginaryComponent().doubleValue());
        double newI = (real() * c.imaginaryComponent().doubleValue() + imag() * c.realComponent().doubleValue());
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
        return result.set(result.realComponent().doubleValue() * v.doubleValue(), result.imaginaryComponent().doubleValue() * v.doubleValue());
    }

    @Override
    public IComplexNumber muli(Number v) {
        return muli(v, this);
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
        double newR = (real() * c.realComponent().doubleValue() + imag() * c.imaginaryComponent().doubleValue()) / d;
        double newI = (imag() * c.realComponent().doubleValue() - real() * c.imaginaryComponent().doubleValue()) / d;
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

    @Override
    public IComplexNumber rdiv(IComplexNumber c) {
        return dup().rdivi(c);
    }

    @Override
    public IComplexNumber rdivi(IComplexNumber c, IComplexNumber result) {
        return result.set(c.div(this));
    }

    @Override
    public IComplexNumber rdivi(IComplexNumber c) {
        return rdivi(c, this);
    }

    @Override
    public IComplexNumber rdivi(Number v, IComplexNumber result) {
        double d = result.realComponent().doubleValue() * result.realComponent().doubleValue() + result.imaginaryComponent().doubleValue() * result.imaginaryComponent().doubleValue();
        return result.set(v.doubleValue() * result.realComponent().doubleValue() / d, -v.doubleValue() * result.imaginaryComponent().doubleValue() / d);
    }

    @Override
    public IComplexNumber rdivi(Number v) {
        return rdivi(v, this);
    }

    @Override
    public IComplexNumber rdiv(Number v) {
        return dup().rdivi(v, this);
    }

    @Override
    public boolean eq(IComplexNumber c) {
        return false;
    }

    @Override
    public boolean ne(IComplexNumber c) {
        return false;
    }

    @Override
    public String toString() {
        return super.toString();
    }

    @Override
    public org.jblas.ComplexDouble set(double real, double imag) {
        return super.set(real, imag);
    }

    @Override
    public double real() {
        return super.real();
    }

    @Override
    public double imag() {
        return super.imag();
    }

    @Override
    public Double realComponent() {
        return super.real();
    }

    @Override
    public Double imaginaryComponent() {
        return super.imag();
    }

    @Override
    public org.jblas.ComplexDouble copy(org.jblas.ComplexDouble other) {
        return super.copy(other);
    }

    /**
     * Add two complex numbers in-place
     *
     * @param c
     * @param result
     */
    @Override
    public org.jblas.ComplexDouble addi(org.jblas.ComplexDouble c, org.jblas.ComplexDouble result) {
        return super.addi(c, result);
    }

    /**
     * Add two complex numbers in-place storing the result in this.
     *
     * @param c
     */
    @Override
    public org.jblas.ComplexDouble addi(org.jblas.ComplexDouble c) {
        return super.addi(c);
    }

    /**
     * Add two complex numbers.
     *
     * @param c
     */
    @Override
    public org.jblas.ComplexDouble add(org.jblas.ComplexDouble c) {
        return super.add(c);
    }

    /**
     * Add a realComponent number to a complex number in-place.
     *
     * @param a
     * @param result
     */
    @Override
    public org.jblas.ComplexDouble addi(double a, org.jblas.ComplexDouble result) {
        return super.addi(a, result);
    }

    /**
     * Add a realComponent number to complex number in-place, storing the result in this.
     *
     * @param c
     */
    @Override
    public org.jblas.ComplexDouble addi(double c) {
        return super.addi(c);
    }

    /**
     * Add a realComponent number to a complex number.
     *
     * @param c
     */
    @Override
    public org.jblas.ComplexDouble add(double c) {
        return super.add(c);
    }

    /**
     * Subtract two complex numbers, in-place
     *
     * @param c
     * @param result
     */
    @Override
    public org.jblas.ComplexDouble subi(org.jblas.ComplexDouble c, org.jblas.ComplexDouble result) {
        return super.subi(c, result);
    }

    @Override
    public org.jblas.ComplexDouble subi(org.jblas.ComplexDouble c) {
        return super.subi(c);
    }

    /**
     * Subtract two complex numbers
     *
     * @param c
     */
    @Override
    public org.jblas.ComplexDouble sub(org.jblas.ComplexDouble c) {
        return super.sub(c);
    }

    @Override
    public org.jblas.ComplexDouble subi(double a, org.jblas.ComplexDouble result) {
        return super.subi(a, result);
    }

    @Override
    public org.jblas.ComplexDouble subi(double a) {
        return super.subi(a);
    }

    @Override
    public org.jblas.ComplexDouble sub(double r) {
        return super.sub(r);
    }

    /**
     * Multiply two complex numbers, inplace
     *
     * @param c
     * @param result
     */
    @Override
    public org.jblas.ComplexDouble muli(org.jblas.ComplexDouble c, org.jblas.ComplexDouble result) {
        return super.muli(c, result);
    }

    @Override
    public org.jblas.ComplexDouble muli(org.jblas.ComplexDouble c) {
        return super.muli(c);
    }

    /**
     * Multiply two complex numbers
     *
     * @param c
     */
    @Override
    public org.jblas.ComplexDouble mul(org.jblas.ComplexDouble c) {
        return super.mul(c);
    }

    @Override
    public org.jblas.ComplexDouble mul(double v) {
        return super.mul(v);
    }

    @Override
    public org.jblas.ComplexDouble muli(double v, org.jblas.ComplexDouble result) {
        return super.muli(v, result);
    }

    @Override
    public org.jblas.ComplexDouble muli(double v) {
        return super.muli(v);
    }

    /**
     * Divide two complex numbers
     *
     * @param c
     */
    @Override
    public ComplexDouble div(org.jblas.ComplexDouble c) {
        return dup().divi(c);

    }

    /**
     * Divide two complex numbers, in-place
     *
     * @param c
     * @param result
     */
    @Override
    public org.jblas.ComplexDouble divi(org.jblas.ComplexDouble c, org.jblas.ComplexDouble result) {
        return super.divi(c, result);
    }

    @Override
    public ComplexDouble divi(org.jblas.ComplexDouble c) {
        super.divi(c);
        return this;
    }

    @Override
    public ComplexDouble divi(double v, org.jblas.ComplexDouble result) {
        super.divi(v, result);
        return this;
    }

    @Override
    public ComplexDouble divi(double v) {
        super.divi(v);
        return this;
    }

    @Override
    public ComplexDouble div(double v) {
        super.div(v);
        return this;
    }

    /**
     * Return the absolute value
     */
    @Override
    public Double absoluteValue() {
        return super.abs();
    }

    /**
     * Returns the argument of a complex number.
     */
    @Override
    public Double complexArgument() {
        return Math.acos(realComponent() / absoluteValue());
    }

    @Override
    public ComplexDouble invi() {
        double d = realComponent() * realComponent() + imaginaryComponent() * imaginaryComponent();
        set(realComponent() / d, -imaginaryComponent() / d);
        return this;
    }

    @Override
    public ComplexDouble inv() {
        return dup().invi();
    }

    @Override
    public IComplexNumber exp() {
        IComplexNumber result = dup();
        double realExp = FastMath.exp(realComponent());
        return result.set(realExp * FastMath.cos(imaginaryComponent()), realExp * FastMath.sin(imaginaryComponent()));
    }

    @Override
    public IComplexNumber log() {
        IComplexNumber result = dup();
        double real = (double) result.realComponent();
        double imaginary = (double) result.imaginaryComponent();
        double modulus = FastMath.sqrt(real*real + imaginary*imaginary);
        double arg = FastMath.atan2(imaginary,real);
        return result.set(FastMath.log(modulus), arg);
    }

    @Override
    public ComplexDouble neg() {
        return dup().negi();
    }

    @Override
    public ComplexDouble negi() {
        set(-realComponent(), -imaginaryComponent());
        return this;
    }

    @Override
    public ComplexDouble sqrt() {
        double a = absoluteValue();
        double s2 = Math.sqrt(2);
        double p = Math.sqrt(a + realComponent()) / s2;
        double q = Math.sqrt(a - realComponent()) / s2 * Math.signum(imaginaryComponent());
        return new ComplexDouble(p, q);
    }

    /**
     * Comparing two floatComplex values.
     *
     * @param o
     */
    @Override
    public boolean equals(Object o) {
        if (!(o instanceof IComplexNumber) || !(o instanceof org.jblas.ComplexFloat) && !(o instanceof org.jblas.ComplexDouble)) {
            return false;
        }
        else {

            IComplexNumber num = (IComplexNumber) o;
            double thisReal = realComponent().doubleValue();
            double otherReal = num.realComponent().doubleValue();
            double thisImag = imaginaryComponent().doubleValue();
            double otherImag = imaginaryComponent().doubleValue();
            double diff = Math.abs(thisReal - otherReal);
            double imagDiff = Math.abs(thisImag - otherImag);
            return diff < Nd4j.EPS_THRESHOLD && imagDiff < Nd4j.EPS_THRESHOLD;
        }
    }

    @Override
    public boolean eq(org.jblas.ComplexDouble c) {
        return super.eq(c);
    }

    @Override
    public boolean ne(org.jblas.ComplexDouble c) {
        return super.ne(c);
    }

    @Override
    public boolean isZero() {
        return super.isZero();
    }

    @Override
    public boolean isReal() {
        return imaginaryComponent() == 0.0;
    }

    @Override
    public boolean isImag() {
        return realComponent() == 0.0;
    }
}
