// --- BEGIN LICENSE BLOCK ---
/* 
 * Copyright (c) 2009, Mikio L. Braun
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 * 
 *     * Neither the name of the Technische Universität Berlin nor the
 *       names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// --- END LICENSE BLOCK ---

package org.deeplearning4j.linalg.jcublas.complex;

import org.deeplearning4j.linalg.api.complex.IComplexDouble;
import org.deeplearning4j.linalg.api.complex.IComplexFloat;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;

/**
 * A complex value with double precision.
 * 
 * @author Mikio L. Braun
 * 
 */
public class ComplexDouble implements IComplexDouble {

    private double r,  i;
    public static final ComplexDouble UNIT = new ComplexDouble(1.0, 0.0);
    public static final ComplexDouble I = new ComplexDouble(0.0, 1.0);
    public static final ComplexDouble NEG_UNIT = new ComplexDouble(-1.0, 0.0);
    public static final ComplexDouble NEG_I = new ComplexDouble(0.0, -1.0);
    public static final ComplexDouble ZERO = new ComplexDouble(0.0);

    public ComplexDouble(double real, double imag) {
        r = real;
        i = imag;
    }

    public ComplexDouble(double real) {
        this(real, 0.0);
    }

    public String toString() {
        if (i >= 0) {
            return r + " + " + i + "i";
        } else {
            return r + " - " + (-i) + "i";
        }
    }

    public ComplexDouble set(double real, double imag) {
        r = real;
        i = imag;
        return this;
    }

    public double real() {
        return r;
    }

    public double imag() {
        return i;
    }

    public ComplexDouble dup() {
        return new ComplexDouble(r, i);
    }

    @Override
    public IComplexNumber copy(IComplexNumber other) {
        return null;
    }

    @Override
    public IComplexNumber addi(IComplexNumber c, IComplexNumber result) {
        return null;
    }

    @Override
    public IComplexNumber addi(IComplexNumber c) {
        return null;
    }

    @Override
    public IComplexNumber add(IComplexNumber c) {
        return null;
    }

    @Override
    public IComplexNumber addi(Number a, IComplexNumber result) {
        return null;
    }

    @Override
    public IComplexNumber addi(Number c) {
        return null;
    }

    @Override
    public IComplexNumber add(Number c) {
        return null;
    }

    @Override
    public IComplexNumber subi(IComplexNumber c, IComplexNumber result) {
        return null;
    }

    @Override
    public IComplexNumber subi(IComplexNumber c) {
        return null;
    }

    @Override
    public IComplexNumber sub(IComplexNumber c) {
        return null;
    }

    @Override
    public IComplexNumber subi(Number a, IComplexNumber result) {
        return null;
    }

    @Override
    public IComplexNumber subi(Number a) {
        return null;
    }

    @Override
    public IComplexNumber sub(Number r) {
        return null;
    }

    @Override
    public IComplexNumber muli(IComplexNumber c, IComplexNumber result) {
        return null;
    }

    @Override
    public IComplexNumber muli(IComplexNumber c) {
        return null;
    }

    @Override
    public IComplexNumber mul(IComplexNumber c) {
        return null;
    }

    @Override
    public IComplexNumber mul(Number v) {
        return null;
    }

    @Override
    public IComplexNumber muli(Number v, IComplexNumber result) {
        return null;
    }

    @Override
    public IComplexNumber muli(Number v) {
        return null;
    }

    @Override
    public IComplexNumber div(IComplexNumber c) {
        return null;
    }

    @Override
    public IComplexNumber divi(IComplexNumber c, IComplexNumber result) {
        return null;
    }

    @Override
    public IComplexNumber divi(IComplexNumber c) {
        return null;
    }

    @Override
    public IComplexNumber divi(Number v, IComplexNumber result) {
        return null;
    }

    @Override
    public IComplexNumber divi(Number v) {
        return null;
    }

    @Override
    public IComplexNumber div(Number v) {
        return null;
    }

    @Override
    public Number absoluteValue() {
        return null;
    }

    @Override
    public Number complexArgument() {
        return null;
    }

    public ComplexDouble copy(ComplexDouble other) {
        r = other.r;
        i = other.i;
        return this;
    }

    /** Add two complex numbers in-place */
    public ComplexDouble addi(ComplexDouble c, ComplexDouble result) {
        if (this == result) {
            r += c.r;
            i += c.i;
        } else {
            result.r = r + c.r;
            result.i = i + c.i;
        }
        return result;
    }

    /** Add two complex numbers in-place storing the result in this. */
    public ComplexDouble addi(ComplexDouble c) {
        return addi(c, this);
    }

    /** Add two complex numbers. */
    public ComplexDouble add(ComplexDouble c) {
        return dup().addi(c);
    }

    /** Add a real number to a complex number in-place. */
    public ComplexDouble addi(double a, ComplexDouble result) {
        if (this == result) {
            r += a;
        } else {
            result.r = r + a;
            result.i = i;
        }
        return result;
    }

    /** Add a real number to complex number in-place, storing the result in this. */
    public ComplexDouble addi(double c) {
        return addi(c, this);
    }

    /** Add a real number to a complex number. */
    public ComplexDouble add(double c) {
        return dup().addi(c);
    }

    /** Subtract two complex numbers, in-place */
    public ComplexDouble subi(ComplexDouble c, ComplexDouble result) {
        if (this == result) {
            r -= c.r;
            i -= c.i;
        } else {
            result.r = r - c.r;
            result.i = i - c.i;
        }
        return this;
    }

    public ComplexDouble subi(ComplexDouble c) {
        return subi(c, this);
    }

    /** Subtract two complex numbers */
    public ComplexDouble sub(ComplexDouble c) {
        return dup().subi(c);
    }

    public ComplexDouble subi(double a, ComplexDouble result) {
        if (this == result) {
            r -= a;
        } else {
            result.r = r - a;
            result.i = i;
        }
        return result;
    }

    public ComplexDouble subi(double a) {
        return subi(a, this);
    }

    public ComplexDouble sub(double r) {
        return dup().subi(r);
    }

    /** Multiply two complex numbers, inplace */
    public ComplexDouble muli(ComplexDouble c, ComplexDouble result) {
        double newR = r * c.r - i * c.i;
        double newI = r * c.i + i * c.r;
        result.r = newR;
        result.i = newI;
        return result;
    }

    public ComplexDouble muli(ComplexDouble c) {
        return muli(c, this);
    }

    /** Multiply two complex numbers */
    public ComplexDouble mul(ComplexDouble c) {
        return dup().muli(c);
    }

    public ComplexDouble mul(double v) {
        return dup().muli(v);
    }

    public ComplexDouble muli(double v, ComplexDouble result) {
        if (this == result) {
            r *= v;
            i *= v;
        } else {
            result.r = r * v;
            result.i = i * v;
        }
        return this;
    }

    public ComplexDouble muli(double v) {
        return muli(v, this);
    }

    /** Divide two complex numbers */
    public ComplexDouble div(ComplexDouble c) {
        return dup().divi(c);
    }

    /** Divide two complex numbers, in-place */
    public ComplexDouble divi(ComplexDouble c, ComplexDouble result) {
        double d = c.r * c.r + c.i * c.i;
        double newR = (r * c.r + i * c.i) / d;
        double newI = (i * c.r - r * c.i) / d;
        result.r = newR;
        result.i = newI;
        return result;
    }

    public ComplexDouble divi(ComplexDouble c) {
        return divi(c, this);
    }

    public ComplexDouble divi(double v, ComplexDouble result) {
        if (this == result) {
            r /= v;
            i /= v;
        } else {
            result.r = r / v;
            result.i = i / v;
        }
        return this;
    }

    public ComplexDouble divi(double v) {
        return divi(v, this);
    }

    public ComplexDouble div(double v) {
        return dup().divi(v);
    }

    /** Return the absolute value */
    public double abs() {
        return (double) Math.sqrt(r * r + i * i);
    }

    /** Returns the argument of a complex number. */
    public double arg() {
        return (double) Math.acos(r/abs());
    }

    public ComplexDouble invi() {
        double d = r * r + i * i;
        r = r / d;
        i = -i / d;
        return this;
    }

    public ComplexDouble inv() {
        return dup().invi();
    }

    public ComplexDouble neg() {
        return dup().negi();
    }

    public ComplexDouble negi() {
        r = -r;
        i = -i;
        return this;
    }

    public ComplexDouble conji() {
        i = -i;
        return this;
    }

    public ComplexDouble conj() {
        return dup().conji();
    }

    public ComplexDouble sqrt() {
        double a = abs();
        double s2 = (double)Math.sqrt(2);
        double p = (double)Math.sqrt(a + r)/s2;
        double q = (double)Math.sqrt(a - r)/s2 * Math.signum(i);
        return new ComplexDouble(p, q);
    }

    @Override
    public boolean eq(IComplexNumber c) {
        return false;
    }

    @Override
    public boolean ne(IComplexNumber c) {
        return false;
    }

    /**
     * Comparing two DoubleComplex values.
     */
    public boolean equals(Object o) {
        if (!(o instanceof ComplexDouble)) {
            return false;
        }
        ComplexDouble c = (ComplexDouble) o;

        return eq(c);
    }

    public boolean eq(ComplexDouble c) {
        return Math.abs(r - c.r) + Math.abs(i - c.i) < (double) 1e-6;
    }

    public boolean ne(ComplexDouble c) {
        return !eq(c);
    }

    public boolean isZero() {
        return r == 0.0 && i == 0.0;
    }
    
    public boolean isReal() {
        return i == 0.0;
    }
    
    public boolean isImag() {
        return r == 0.0;
    }

    @Override
    public IComplexFloat asFloat() {
        return null;
    }

    @Override
    public IComplexDouble asDouble() {
        return null;
    }

    @Override
    public IComplexNumber set(Number real, Number imag) {
        return null;
    }

    @Override
    public Double realComponent() {
        return null;
    }

    @Override
    public Double imaginaryComponent() {
        return null;
    }
}

