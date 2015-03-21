/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.complex;

import org.nd4j.linalg.factory.Nd4j;

/**
 * Base complex float class
 *
 * @author Adam Gibson
 */
public abstract class BaseComplexFloat implements IComplexFloat {

    protected float real, imag;

    public BaseComplexFloat() {
    }

    public BaseComplexFloat(Float real, Float imag) {
        this.real = real;
        this.imag = imag;
    }

    public BaseComplexFloat(float real, float imag) {
        this.real = real;
        this.imag = imag;

    }

    public BaseComplexFloat(float real) {
        this(real, 0);
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
        return set(realComponent().floatValue(), realComponent().floatValue());
    }

    @Override
    public IComplexNumber rsubi(IComplexNumber a, IComplexNumber result) {
        return result.set(a.sub(this));
    }

    @Override
    public IComplexNumber rsub(IComplexNumber c) {
        return dup().rsubi(c);
    }

    @Override
    public IComplexNumber rsubi(Number a, IComplexNumber result) {
        return result.set(a.doubleValue() - realComponent().doubleValue(), imaginaryComponent());
    }

    @Override
    public IComplexNumber rsubi(Number a) {
        return rsubi(a, this);
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
        return result.set(c.div(this));
    }

    @Override
    public IComplexNumber rdivi(IComplexNumber c) {
        return rdivi(c, this);
    }

    @Override
    public IComplexNumber rdivi(Number v, IComplexNumber result) {
        return null;
    }

    @Override
    public IComplexNumber rdivi(Number v) {
        return rdivi(v, this);
    }

    @Override
    public IComplexNumber rdiv(Number v) {
        return dup().rdivi(v);
    }

    /**
     * Convert to a double
     *
     * @return this complex number as a double
     */
    @Override
    public IComplexDouble asDouble() {
        return Nd4j.createDouble(realComponent(), imaginaryComponent());
    }

    /**
     * Convert to a float
     *
     * @return this complex number as a float
     */
    @Override
    public IComplexFloat asFloat() {
        return this;
    }


    @Override
    public IComplexFloat conji() {
        set(realComponent(), -imaginaryComponent());
        return this;
    }

    @Override
    public IComplexNumber conj() {
        return dup().conji();
    }

    @Override
    public IComplexNumber set(Number real, Number imag) {
        this.real = real.floatValue();
        this.imag = imag.floatValue();
        return this;
    }

    @Override
    public IComplexNumber copy(IComplexNumber other) {

        return Nd4j.createFloat(other.realComponent().floatValue(), other.imaginaryComponent().floatValue());

    }

    /**
     * Add two complex numbers in-place
     *
     * @param c
     * @param result
     */
    @Override
    public IComplexNumber addi(IComplexNumber c, IComplexNumber result) {
        if (this == result) {
            set(realComponent() + c.realComponent().floatValue(), imaginaryComponent() + result.imaginaryComponent().floatValue());
        } else {
            result.set(result.realComponent().floatValue() + c.realComponent().floatValue(), result.imaginaryComponent().floatValue() + c.imaginaryComponent().floatValue());

        }
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
        if (this == result) {
            set(realComponent() + a.floatValue(), imaginaryComponent() + a.floatValue());
        } else {
            result.set(result.realComponent().floatValue() + a.floatValue(), imaginaryComponent() + a.floatValue());

        }
        return result;
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
        if (this == result) {
            set(realComponent() - c.realComponent().floatValue(), imaginaryComponent() - result.imaginaryComponent().floatValue());
        } else {
            return result.set(result.realComponent().floatValue() - c.realComponent().floatValue(), result.imaginaryComponent().floatValue() - c.imaginaryComponent().floatValue());

        }
        return this;
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
        if (this == result) {
            set(realComponent() - a.floatValue(), imaginaryComponent() - a.floatValue());
        } else {
            return result.set(result.realComponent().floatValue() - a.floatValue(), imaginaryComponent() - a.floatValue());

        }
        return result;
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
        float newR = realComponent() * c.realComponent().floatValue() - imaginaryComponent() * c.imaginaryComponent().floatValue();
        float newI = realComponent() * c.imaginaryComponent().floatValue() + imaginaryComponent() * c.realComponent().floatValue();
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
        if (this == result) {
            set(realComponent() * v.floatValue(), imaginaryComponent() * v.floatValue());
        } else {
            result.set(result.realComponent().floatValue() * v.floatValue(), result.imaginaryComponent().floatValue() * v.floatValue());

        }
        return result;
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
        float d = c.realComponent().floatValue() * c.realComponent().floatValue() + c.imaginaryComponent().floatValue() * c.imaginaryComponent().floatValue();
        float newR = (realComponent() * c.realComponent().floatValue() + imaginaryComponent() * c.imaginaryComponent().floatValue()) / d;
        float newI = (imaginaryComponent() * c.realComponent().floatValue() - realComponent() * c.imaginaryComponent().floatValue()) / d;
        result.set(newR, newI);
        return result;
    }

    @Override
    public IComplexNumber divi(IComplexNumber c) {
        return divi(c, this);
    }

    @Override
    public IComplexNumber divi(Number v, IComplexNumber result) {
        if (this == result) {
            set(realComponent() / v.floatValue(), imaginaryComponent() / v.floatValue());
        } else {
            result.set(result.realComponent().floatValue() / v.floatValue(), imaginaryComponent() / v.floatValue());

        }
        return result;
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
        return false;
    }

    @Override
    public boolean ne(IComplexNumber c) {
        return false;
    }


    @Override
    public Float realComponent() {
        return real;
    }

    @Override
    public Float imaginaryComponent() {
        return imag;
    }


    @Override
    public IComplexFloat divi(float v) {
        this.real = real / v;
        this.imag = imag / v;
        return this;
    }

    @Override
    public IComplexNumber div(float v) {
        return dup().divi(v);
    }

    /**
     * Return the absolute value
     */
    @Override
    public Float absoluteValue() {
        return (float) Math.sqrt(real * real + imag * imag);
    }

    /**
     * Returns the argument of a complex number.
     */
    @Override
    public Float complexArgument() {
        return (float) Math.acos(realComponent() / absoluteValue());
    }

    @Override
    public IComplexFloat invi() {
        float d = realComponent() * realComponent() + imaginaryComponent() * imaginaryComponent();
        set(realComponent() / d, -imaginaryComponent() / d);
        return this;
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
    public IComplexFloat negi() {
        set(-realComponent(), -imaginaryComponent());
        return this;
    }

    @Override
    public IComplexFloat sqrt() {
        float a = absoluteValue();
        float s2 = (float) Math.sqrt(2);
        float p = (float) Math.sqrt(a + realComponent()) / s2;
        float q = (float) Math.sqrt(a - realComponent()) / s2 * Math.signum(imaginaryComponent());
        return Nd4j.createFloat(p, q);
    }

    /**
     * Comparing two floatComplex values.
     *
     * @param o
     */
    @Override
    public boolean equals(Object o) {
        return super.equals(o);
    }

    public boolean isZero() {
        return real == 0;
    }

    @Override
    public boolean isReal() {
        return imaginaryComponent() == 0;
    }


    @Override
    public boolean isImag() {
        return realComponent() == 0;
    }

    @Override
    public int hashCode() {
        int result = (real != +0.0f ? Float.floatToIntBits(real) : 0);
        result = 31 * result + (imag != +0.0f ? Float.floatToIntBits(imag) : 0);
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
