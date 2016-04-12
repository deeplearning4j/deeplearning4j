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

package org.nd4j.linalg.util;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class ComplexUtil {
    private ComplexUtil() {
    }
    /**
     * Create complex number where the
     * @param realComponents the real components for the complex
     * @return the complex numbers based on the given real components
     */
    public static IComplexNumber[][] complexNumbersFor(float[][] realComponents) {
        IComplexNumber[][] ret = new IComplexNumber[realComponents.length][realComponents[0].length];
        for(int i = 0; i < realComponents.length; i++)
            for(int j = 0; j < realComponents[i].length; j++)
                ret[i][j] = Nd4j.createComplexNumber(realComponents[i][j],0);
        return ret;
    }

    /**
     * Create complex number where the
     * @param realComponents the real components for the complex
     * @return the complex numbers based on the given real components
     */
    public static IComplexNumber[][] complexNumbersFor(double[][] realComponents) {
        IComplexNumber[][] ret = new IComplexNumber[realComponents.length][realComponents[0].length];
        for(int i = 0; i < realComponents.length; i++)
            for(int j = 0; j < realComponents[i].length; j++)
                ret[i][j] = Nd4j.createComplexNumber(realComponents[i][j],0);
        return ret;
    }

    /**
     * Create complex number where the
     * @param realComponents the real components for the complex
     * @return the complex numbers based on the given real components
     */
    public static IComplexNumber[] complexNumbersFor(float[] realComponents) {
        IComplexNumber[] ret = new IComplexNumber[realComponents.length];
        for(int i = 0; i < realComponents.length; i++)
            ret[i] = Nd4j.createComplexNumber(realComponents[i],0);
        return ret;
    }

    /**
     * Create complex number where the
     * @param realComponents the real components for the complex
     * @return the complex numbers based on the given real components
     */
    public static IComplexNumber[] complexNumbersFor(double[] realComponents) {
        IComplexNumber[] ret = new IComplexNumber[realComponents.length];
        for(int i = 0; i < realComponents.length; i++)
            ret[i] = Nd4j.createComplexNumber(realComponents[i],0);
        return ret;
    }

    /**
     * Return the  sin value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber atan(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).atan();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }

    /**
     * Return the  sin value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber acos(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).acos();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }

    /**
     * Return the  sin value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber asin(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).asin();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }

    /**
     * Return the  sin value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber sin(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).sin();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }

    /**
     * Return the  ceiling value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber ceil(IComplexNumber num) {
        Complex c = new Complex(FastMath.ceil(num.realComponent().doubleValue()), FastMath.ceil(num.imaginaryComponent().doubleValue()));
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }

    /**
     * Return the  floor value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber floor(IComplexNumber num) {
        Complex c = new Complex(FastMath.floor(num.realComponent().doubleValue()), FastMath.floor(num.imaginaryComponent().doubleValue()));
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }


    /**
     * Return the  log value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber neg(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).negate();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }


    /**
     * Return the  log value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber log(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).log();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }


    /**
     * Return the absolute value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber sqrt(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).sqrt();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }


    /**
     * Return the absolute value of the given complex number
     *
     * @param num the number to getScalar the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber abs(IComplexNumber num) {
        double c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).abs();
        return Nd4j.createDouble(c, 0);
    }


    public static IComplexNumber round(IComplexNumber num) {
        return Nd4j.createDouble(Math.round(num.realComponent().doubleValue()), Math.round(num.imaginaryComponent().doubleValue()));
    }

    /**
     * Raise a complex number to a power
     *
     * @param num   the number to raise
     * @param power the power to raise to
     * @return the number raised to a power
     */
    public static IComplexNumber pow(IComplexNumber num, IComplexNumber power) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).pow(new Complex(power.realComponent().doubleValue(), power.imaginaryComponent().doubleValue()));
        if (c.isNaN())
            c = new Complex(Nd4j.EPS_THRESHOLD, 0.0);
        return Nd4j.createDouble(c.getReal(), c.getImaginary());

    }

    /**
     * Raise a complex number to a power
     *
     * @param num   the number to raise
     * @param power the power to raise to
     * @return the number raised to a power
     */
    public static IComplexNumber pow(IComplexNumber num, double power) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).pow(power);
        if (c.isNaN())
            c = new Complex(Nd4j.EPS_THRESHOLD, 0.0);
        return Nd4j.createDouble(c.getReal(), c.getImaginary());

    }

    /**
     * Return the cos of a complex number
     *
     * @param num the tanh of a complex number
     * @return the tanh of a complex number
     */
    public static IComplexNumber cos(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).cos();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }

    /**
     * Return the tanh of a complex number
     *
     * @param num the tanh of a complex number
     * @return the tanh of a complex number
     */
    public static IComplexNumber hardTanh(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).tanh();
        if (c.getReal() < -1.0)
            c = new Complex(-1.0, c.getImaginary());
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }

    /**
     * Return the tanh of a complex number
     *
     * @param num the tanh of a complex number
     * @return the tanh of a complex number
     */
    public static IComplexNumber tanh(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue()).tanh();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }


    /**
     * Returns the exp of a complex number:
     * Let r be the realComponent component and i be the imaginary
     * Let ret be the complex number returned
     * ret -> exp(r) * cos(i), exp(r) * sin(i)
     * where the first number is the realComponent component
     * and the second number is the imaginary component
     *
     * @param d the number to getFromOrigin the exp of
     * @return the exponential of this complex number
     */
    public static IComplexNumber exp(IComplexNumber d) {
        if (d instanceof IComplexFloat)
            return exp((IComplexFloat) d);
        return exp((IComplexDouble) d);
    }


    /**
     * Returns the exp of a complex number:
     * Let r be the realComponent component and i be the imaginary
     * Let ret be the complex number returned
     * ret -> exp(r) * cos(i), exp(r) * sin(i)
     * where the first number is the realComponent component
     * and the second number is the imaginary component
     *
     * @param d the number to getFromOrigin the exp of
     * @return the exponential of this complex number
     */
    public static IComplexDouble exp(IComplexDouble d) {
        return Nd4j.createDouble(FastMath.exp(d.realComponent()) * FastMath.cos(d.imaginaryComponent()), FastMath.exp(d.realComponent()) * FastMath.sin(d.imaginaryComponent()));
    }

    /**
     * Returns the exp of a complex number:
     * Let r be the realComponent component and i be the imaginary
     * Let ret be the complex number returned
     * ret -> exp(r) * cos(i), exp(r) * sin(i)
     * where the first number is the realComponent component
     * and the second number is the imaginary component
     *
     * @param d the number to getFromOrigin the exp of
     * @return the exponential of this complex number
     */
    public static IComplexFloat exp(IComplexFloat d) {
        return Nd4j.createFloat((float) FastMath.exp(d.realComponent()) * (float) FastMath.cos(d.imaginaryComponent()), (float) FastMath.exp(d.realComponent()) * (float) FastMath.sin(d.imaginaryComponent()));
    }


}
