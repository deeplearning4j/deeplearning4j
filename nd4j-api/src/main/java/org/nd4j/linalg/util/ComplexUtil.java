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



    /**
     * Return the  log value of the given complex number
     * @param num the number to get the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber floor(IComplexNumber num) {
        Complex c = new Complex(Math.floor(num.realComponent().doubleValue()),Math.floor(num.imaginaryComponent().doubleValue()));
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }



    /**
     * Return the  log value of the given complex number
     * @param num the number to get the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber neg(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(),num.imaginaryComponent().doubleValue()).negate();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }



    /**
     * Return the  log value of the given complex number
     * @param num the number to get the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber log(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(),num.imaginaryComponent().doubleValue()).log();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }


    /**
     * Return the absolute value of the given complex number
     * @param num the number to get the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber sqrt(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(),num.imaginaryComponent().doubleValue()).sqrt();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }


    /**
     * Return the absolute value of the given complex number
     * @param num the number to get the absolute value for
     * @return the absolute value of this complex number
     */
    public static IComplexNumber abs(IComplexNumber num) {
        double c = new Complex(num.realComponent().doubleValue(),num.imaginaryComponent().doubleValue()).abs();
        return Nd4j.createDouble(c, 0);
    }


    public static IComplexNumber round(IComplexNumber num) {
        return Nd4j.createDouble(Math.round(num.realComponent().doubleValue()), Math.round(num.realComponent().doubleValue()));
    }

    /**
     * Raise a complex number to a power
     * @param num the number to raise
     * @param power the power to raise to
     * @return the number raised to a power
     */
    public static IComplexNumber pow(IComplexNumber num,IComplexNumber power) {
        Complex c = new Complex(num.realComponent().doubleValue(),num.imaginaryComponent().doubleValue()).pow(new Complex(power.realComponent().doubleValue(),power.imaginaryComponent().doubleValue()));
        return Nd4j.createDouble(c.getReal(), c.getImaginary());

    }

    /**
     * Raise a complex number to a power
     * @param num the number to raise
     * @param power the power to raise to
     * @return the number raised to a power
     */
    public static IComplexNumber pow(IComplexNumber num,double power) {
        Complex c = new Complex(num.realComponent().doubleValue(),num.imaginaryComponent().doubleValue()).pow(power);
        return Nd4j.createDouble(c.getReal(), c.getImaginary());

    }


    /**
     * Return the tanh of a complex number
     * @param num the tanh of a complex number
     * @return the tanh of a complex number
     */
    public static IComplexNumber tanh(IComplexNumber num) {
        Complex c = new Complex(num.realComponent().doubleValue(),num.imaginaryComponent().doubleValue()).tanh();
        return Nd4j.createDouble(c.getReal(), c.getImaginary());
    }



    /**
     * Returns the exp of a complex number:
     * Let r be the realComponent component and i be the imaginary
     * Let ret be the complex number returned
     * ret -> exp(r) * cos(i), exp(r) * sin(i)
     * where the first number is the realComponent component
     * and the second number is the imaginary component
     * @param d the number to getFromOrigin the exp of
     * @return the exponential of this complex number
     */
    public static IComplexNumber exp(IComplexNumber d) {
        if(d instanceof  IComplexFloat)
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
     * @param d the number to getFromOrigin the exp of
     * @return the exponential of this complex number
     */
    public static IComplexDouble exp(IComplexDouble d) {
        return  Nd4j.createDouble(FastMath.exp(d.realComponent()) * FastMath.cos(d.imaginaryComponent()), FastMath.exp(d.realComponent()) * FastMath.sin(d.imaginaryComponent()));
    }

    /**
     * Returns the exp of a complex number:
     * Let r be the realComponent component and i be the imaginary
     * Let ret be the complex number returned
     * ret -> exp(r) * cos(i), exp(r) * sin(i)
     * where the first number is the realComponent component
     * and the second number is the imaginary component
     * @param d the number to getFromOrigin the exp of
     * @return the exponential of this complex number
     */
    public static IComplexFloat exp(IComplexFloat d) {
        return  Nd4j.createFloat((float) FastMath.exp(d.realComponent()) * (float) FastMath.cos(d.imaginaryComponent()), (float) FastMath.exp(d.realComponent()) * (float) FastMath.sin(d.imaginaryComponent()));
    }



}
