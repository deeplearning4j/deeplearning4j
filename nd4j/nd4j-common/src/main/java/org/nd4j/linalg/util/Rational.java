/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.util;
/*
 * To change this template, choose Tools | Templates and open the template in the editor.
 */

// package org.nevec.rjm ;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;

/**
 * Fractions (rational numbers).
 * They are divisions of two BigInteger variables, reduced to greatest
 * common divisors of 1.
 */
class Rational implements Cloneable {

    /* The maximum and minimum value of a standard Java integer, 2^31.
     */
    static BigInteger MAX_INT = BigInteger.valueOf(Integer.MAX_VALUE);
    static BigInteger MIN_INT = BigInteger.valueOf(Integer.MIN_VALUE);
    static Rational ONE = new Rational(1, 1);
    static Rational ZERO = new Rational();
    /**
     * numerator
     */
    BigInteger a;
    /**
     * denominator
     */
    BigInteger b;

    /**
     * Default ctor, which represents the zero.
     */
    public Rational() {
        a = BigInteger.ZERO;
        b = BigInteger.ONE;
    }

    /**
     * ctor from a numerator and denominator.
     *
     * @param a the numerator.
     * @param b the denominator.
     */
    public Rational(BigInteger a, BigInteger b) {
        this.a = a;
        this.b = b;
        normalize();
    }

    /**
     * ctor from a numerator.
     *
     * @param a the BigInteger.
     */
    public Rational(BigInteger a) {
        this.a = a;
        b = BigInteger.valueOf(1);
    }

    /**
     * ctor from a numerator and denominator.
     *
     * @param a the numerator.
     * @param b the denominator.
     */
    public Rational(int a, int b) {
        this(BigInteger.valueOf(a), BigInteger.valueOf(b));
    }

    /**
     * ctor from a string representation.
     *
     * @param str the string.
     *            This either has a slash in it, separating two integers, or, if there is no slash,
     *            is representing the numerator with implicit denominator equal to 1.
     * @warning this does not yet test for a denominator equal to zero
     */
    public Rational(String str) throws NumberFormatException {
        this(str, 10);
    }

    /**
     * ctor from a string representation in a specified base.
     *
     * @param str   the string.
     *              This either has a slash in it, separating two integers, or, if there is no slash,
     *              is just representing the numerator.
     * @param radix the number base for numerator and denominator
     * @warning this does not yet test for a denominator equal to zero
     * 5
     */
    public Rational(String str, int radix) throws NumberFormatException {
        int hasslah = str.indexOf("/");
        if (hasslah == -1) {
            a = new BigInteger(str, radix);
            b = new BigInteger("1", radix);
            /* no normalization necessary here */
        } else {
            /* create numerator and denominator separately
             */
            a = new BigInteger(str.substring(0, hasslah), radix);
            b = new BigInteger(str.substring(hasslah + 1), radix);
            normalize();
        }
    }

    /**
     * binomial (n choose m).
     *
     * @param n the numerator. Equals the size of the set to choose from.
     * @param m the denominator. Equals the number of elements to select.
     * @return the binomial coefficient.
     */
    public static Rational binomial(Rational n, BigInteger m) {
        if (m.compareTo(BigInteger.ZERO) == 0) {
            return Rational.ONE;
        }
        Rational bin = n;
        for (BigInteger i = BigInteger.valueOf(2); i.compareTo(m) != 1; i = i.add(BigInteger.ONE)) {
            bin = bin.multiply(n.subtract(i.subtract(BigInteger.ONE))).divide(i);
        }
        return bin;
    } /* Rational.binomial */

    /**
     * binomial (n choose m).
     *
     * @param n the numerator. Equals the size of the set to choose from.
     * @param m the denominator. Equals the number of elements to select.
     * @return the binomial coefficient.
     */
    public static Rational binomial(Rational n, int m) {
        if (m == 0) {
            return Rational.ONE;
        }
        Rational bin = n;
        for (int i = 2; i <= m; i++) {
            bin = bin.multiply(n.subtract(i - 1)).divide(i);
        }
        return bin;
    } /* Rational.binomial */

    /**
     * Create a copy.
     */
    @Override
    public Rational clone() {
        /* protected access means this does not work
         * return new Rational(a.clone(), b.clone()) ;
         */
        BigInteger aclon = new BigInteger("" + a);
        BigInteger bclon = new BigInteger("" + b);
        return new Rational(aclon, bclon);
    } /* Rational.clone */

    /**
     * Multiply by another fraction.
     *
     * @param val a second rational number.
     * @return the product of this with the val.
     */
    public Rational multiply(final Rational val) {
        BigInteger num = a.multiply(val.a);
        BigInteger deno = b.multiply(val.b);
        /* Normalization to an coprime format will be done inside
         * the ctor() and is not duplicated here.
         */
        return (new Rational(num, deno));
    } /* Rational.multiply */

    /**
     * Multiply by a BigInteger.
     *
     * @param val a second number.
     * @return the product of this with the value.
     */
    public Rational multiply(final BigInteger val) {
        Rational val2 = new Rational(val, BigInteger.ONE);
        return (multiply(val2));
    } /* Rational.multiply */

    /**
     * Multiply by an integer.
     *
     * @param val a second number.
     * @return the product of this with the value.
     */
    public Rational multiply(final int val) {
        BigInteger tmp = BigInteger.valueOf(val);
        return multiply(tmp);
    } /* Rational.multiply */

    /**
     * Power to an integer.
     *
     * @param exponent the exponent.
     * @return this value raised to the power given by the exponent.
     * If the exponent is 0, the value 1 is returned.
     */
    public Rational pow(int exponent) {
        if (exponent == 0) {
            return new Rational(1, 1);
        }
        BigInteger num = a.pow(Math.abs(exponent));
        BigInteger deno = b.pow(Math.abs(exponent));
        if (exponent > 0) {
            return (new Rational(num, deno));
        } else {
            return (new Rational(deno, num));
        }
    } /* Rational.pow */

    /**
     * Power to an integer.
     *
     * @param exponent the exponent.
     * @return this value raised to the power given by the exponent.
     * If the exponent is 0, the value 1 is returned.
     */
    public Rational pow(BigInteger exponent) throws NumberFormatException {
        /* test for overflow */
        if (exponent.compareTo(MAX_INT) == 1) {
            throw new NumberFormatException("Exponent " + exponent.toString() + " too large.");
        }
        if (exponent.compareTo(MIN_INT) == -1) {
            throw new NumberFormatException("Exponent " + exponent.toString() + " too small.");
        }
        /* promote to the simpler interface above */
        return pow(exponent.intValue());
    } /* Rational.pow */

    /**
     * Divide by another fraction.
     *
     * @param val A second rational number.
     * @return The value of this/val
     */
    public Rational divide(final Rational val) {
        BigInteger num = a.multiply(val.b);
        BigInteger deno = b.multiply(val.a);
        /* Reduction to a coprime format is done inside the ctor,
         * and not repeated here.
         */
        return (new Rational(num, deno));
    } /* Rational.divide */

    /**
     * Divide by an integer.
     *
     * @param val a second number.
     * @return the value of this/val
     */
    public Rational divide(BigInteger val) {
        Rational val2 = new Rational(val, BigInteger.ONE);
        return (divide(val2));
    } /* Rational.divide */

    /**
     * Divide by an integer.
     *
     * @param val A second number.
     * @return The value of this/val
     */
    public Rational divide(int val) {
        Rational val2 = new Rational(val, 1);
        return (divide(val2));
    } /* Rational.divide */

    /**
     * Add another fraction.
     *
     * @param val The number to be added
     * @return this+val.
     */
    public Rational add(Rational val) {
        BigInteger num = a.multiply(val.b).add(b.multiply(val.a));
        BigInteger deno = b.multiply(val.b);
        return (new Rational(num, deno));
    } /* Rational.add */

    /**
     * Add another integer.
     *
     * @param val The number to be added
     * @return this+val.
     */
    public Rational add(BigInteger val) {
        Rational val2 = new Rational(val, BigInteger.ONE);
        return (add(val2));
    } /* Rational.add */

    /**
     * Compute the negative.
     *
     * @return -this.
     */
    public Rational negate() {
        return (new Rational(a.negate(), b));
    } /* Rational.negate */

    /**
     * Subtract another fraction.
     * 7
     *
     * @param val the number to be subtracted from this
     * @return this - val.
     */
    public Rational subtract(Rational val) {
        Rational val2 = val.negate();
        return (add(val2));
    } /* Rational.subtract */

    /**
     * Subtract an integer.
     *
     * @param val the number to be subtracted from this
     * @return this - val.
     */
    public Rational subtract(BigInteger val) {
        Rational val2 = new Rational(val, BigInteger.ONE);
        return (subtract(val2));
    } /* Rational.subtract */

    /**
     * Subtract an integer.
     *
     * @param val the number to be subtracted from this
     * @return this - val.
     */
    public Rational subtract(int val) {
        Rational val2 = new Rational(val, 1);
        return (subtract(val2));
    } /* Rational.subtract */

    /**
     * Get the numerator.
     *
     * @return The numerator of the reduced fraction.
     */
    public BigInteger numer() {
        return a;
    }

    /**
     * Get the denominator.
     *
     * @return The denominator of the reduced fraction.
     */
    public BigInteger denom() {
        return b;
    }

    /**
     * Absolute value.
     *
     * @return The absolute (non-negative) value of this.
     */
    public Rational abs() {
        return (new Rational(a.abs(), b.abs()));
    }

    /**
     * floor(): the nearest integer not greater than this.
     *
     * @return The integer rounded towards negative infinity.
     */
    public BigInteger floor() {
        /* is already integer: return the numerator
         */
        if (b.compareTo(BigInteger.ONE) == 0) {
            return a;
        } else if (a.compareTo(BigInteger.ZERO) > 0) {
            return a.divide(b);
        } else {
            return a.divide(b).subtract(BigInteger.ONE);
        }
    } /* Rational.floor */


    /**
     * Remove the fractional part.
     *
     * @return The integer rounded towards zero.
     */
    public BigInteger trunc() {
        /* is already integer: return the numerator
         */
        if (b.compareTo(BigInteger.ONE) == 0) {
            return a;
        } else {
            return a.divide(b);
        }
    } /* Rational.trunc */


    /**
     * Compares the value of this with another constant.
     *
     * @param val the other constant to compare with
     * @return -1, 0 or 1 if this number is numerically less than, equal to,
     * or greater than val.
     */
    public int compareTo(final Rational val) {
        /* Since we have always kept the denominators positive,
         * simple cross-multiplying works without changing the sign.
         */
        final BigInteger left = a.multiply(val.b);
        final BigInteger right = val.a.multiply(b);
        return left.compareTo(right);
    } /* Rational.compareTo */


    /**
     * Compares the value of this with another constant.
     *
     * @param val the other constant to compare with
     * @return -1, 0 or 1 if this number is numerically less than, equal to,
     * or greater than val.
     */
    public int compareTo(final BigInteger val) {
        final Rational val2 = new Rational(val, BigInteger.ONE);
        return (compareTo(val2));
    } /* Rational.compareTo */


    /**
     * Return a string in the format number/denom.
     * If the denominator equals 1, print just the numerator without a slash.
     *
     * @return the human-readable version in base 10
     */
    @Override
    public String toString() {
        if (b.compareTo(BigInteger.ONE) != 0) {
            return (a.toString() + "/" + b.toString());
        } else {
            return a.toString();
        }
    } /* Rational.toString */


    /**
     * Return a double value representation.
     *
     * @return The value with double precision.
     */
    public double doubleValue() {
        /* To meet the risk of individual overflows of the exponents of
         * a separate invocation a.doubleValue() or b.doubleValue(), we divide first
         * in a BigDecimal environment and converst the result.
         */
        BigDecimal adivb = (new BigDecimal(a)).divide(new BigDecimal(b), MathContext.DECIMAL128);
        return adivb.doubleValue();
    } /* Rational.doubleValue */


    /**
     * Return a float value representation.
     *
     * @return The value with single precision.
     */
    public float floatValue() {
        BigDecimal adivb = (new BigDecimal(a)).divide(new BigDecimal(b), MathContext.DECIMAL128);
        return adivb.floatValue();
    } /* Rational.floatValue */


    /**
     * Return a representation as BigDecimal.
     *
     * @param mc the mathematical context which determines precision, rounding mode etc
     * @return A representation as a BigDecimal floating point number.
     */
    public BigDecimal BigDecimalValue(MathContext mc) {
        /* numerator and denominator individually rephrased
         */
        BigDecimal n = new BigDecimal(a);
        BigDecimal d = new BigDecimal(b);
        return n.divide(d, mc);
    } /* Rational.BigDecimnalValue */


    /**
     * Return a string in floating point format.
     *
     * @param digits The precision (number of digits)
     * @return The human-readable version in base 10.
     */
    public String toFString(int digits) {
        if (b.compareTo(BigInteger.ONE) != 0) {
            MathContext mc = new MathContext(digits, RoundingMode.DOWN);
            BigDecimal f = (new BigDecimal(a)).divide(new BigDecimal(b), mc);
            return (f.toString());
        } else {
            return a.toString();
        }
    } /* Rational.toFString */


    /**
     * Compares the value of this with another constant.
     *
     * @param val The other constant to compare with
     * @return The arithmetic maximum of this and val.
     */
    public Rational max(final Rational val) {
        if (compareTo(val) > 0) {
            return this;
        } else {
            return val;
        }
    } /* Rational.max */


    /**
     * Compares the value of this with another constant.
     *
     * @param val The other constant to compare with
     * @return The arithmetic minimum of this and val.
     */
    public Rational min(final Rational val) {
        if (compareTo(val) < 0) {
            return this;
        } else {
            return val;
        }
    } /* Rational.min */


    /**
     * Compute Pochhammer’s symbol (this)_n.
     *
     * @param n The number of product terms in the evaluation.
     * @return Gamma(this+n)/Gamma(this) = this*(this+1)*...*(this+n-1).
     */
    public Rational Pochhammer(final BigInteger n) {
        if (n.compareTo(BigInteger.ZERO) < 0) {
            return null;
        } else if (n.compareTo(BigInteger.ZERO) == 0) {
            return Rational.ONE;
        } else {
            /* initialize results with the current value
             */
            Rational res = new Rational(a, b);
            BigInteger i = BigInteger.ONE;
            for (; i.compareTo(n) < 0; i = i.add(BigInteger.ONE)) {
                res = res.multiply(add(i));
            }
            return res;
        }
    } /* Rational.pochhammer */


    /**
     * Compute pochhammer’s symbol (this)_n.
     *
     * @param n The number of product terms in the evaluation.
     * @return Gamma(this+n)/GAMMA(this).
     */
    public Rational Pochhammer(int n) {
        return Pochhammer(BigInteger.valueOf(n));
    } /* Rational.pochhammer */


    /**
     * Normalize to coprime numerator and denominator.
     * Also copy a negative sign of the denominator to the numerator.
     */
    protected void normalize() {
        /* compute greatest common divisor of numerator and denominator
         */
        final BigInteger g = a.gcd(b);
        if (g.compareTo(BigInteger.ONE) > 0) {
            a = a.divide(g);
            b = b.divide(g);
        }
        if (b.compareTo(BigInteger.ZERO) == -1) {
            a = a.negate();
            b = b.negate();
        }
    } /* Rational.normalize */

} /* Rational */
