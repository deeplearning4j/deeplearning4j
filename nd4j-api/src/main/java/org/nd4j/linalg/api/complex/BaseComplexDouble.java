package org.nd4j.linalg.api.complex;

import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 9/5/14.
 */
public abstract class BaseComplexDouble implements IComplexDouble {
    protected static double real,imag;



    public BaseComplexDouble(double real, double imag) {
        this.real = real;
        this.imag = imag;
    }

    public BaseComplexDouble(double real) {
        this(real,imag);
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

    /**
     * Add two complex numbers in-place
     *
     * @param c
     * @param result
     */
    @Override
    public IComplexNumber addi(IComplexNumber c, IComplexNumber result) {
        if (this == result) {
            set(realComponent() + c.realComponent().doubleValue(),imaginaryComponent() + c.imaginaryComponent().doubleValue());
        } else {
            result.set(result.realComponent().doubleValue() + c.realComponent().doubleValue(),
                    result.imaginaryComponent().doubleValue() + c.imaginaryComponent().doubleValue());

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
        return addi(c,this);
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
            set(realComponent() + a.doubleValue(),imaginaryComponent() + a.doubleValue());
        } else {
            result.set(result.realComponent().doubleValue() + a.doubleValue(),imaginaryComponent() + a.doubleValue());

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
        return addi(c,this);
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
            set(realComponent() - c.realComponent().doubleValue(),imaginaryComponent() - c.imaginaryComponent().doubleValue());
        } else {
            result.set(result.realComponent().doubleValue() - c.realComponent().doubleValue(),result.imaginaryComponent().doubleValue() - c.imaginaryComponent().doubleValue());

        }
        return this;
    }

    @Override
    public IComplexNumber subi(IComplexNumber c) {
        return subi(c,this);
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
            set(realComponent() - a.doubleValue(),imaginaryComponent() - a.doubleValue());
        } else {
            result.set(result.realComponent().doubleValue() - a.doubleValue(),imaginaryComponent() - a.doubleValue());

        }
        return result;
    }

    @Override
    public IComplexNumber subi(Number a) {
        return subi(a,this);
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
        result.set(newR,newI);
        return result;
    }

    @Override
    public IComplexNumber muli(IComplexNumber c) {
        return muli(c,this);
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
            set(realComponent() * v.doubleValue(),imaginaryComponent() * v.doubleValue());
        } else {
            result.set(result.realComponent().doubleValue() * v.doubleValue(),imaginaryComponent() * v.doubleValue());

        }
        return result;
    }

    @Override
    public IComplexNumber muli(Number v) {
        return muli(v,this);
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
        result.set(newR,newI);
        return result;
    }

    @Override
    public IComplexNumber divi(IComplexNumber c) {
        return divi(c,this);
    }

    @Override
    public IComplexNumber divi(Number v, IComplexNumber result) {
        if (this == result) {
            set(realComponent() / v.doubleValue(),imaginaryComponent() / v.doubleValue());
        } else {
            result.set(result.realComponent().doubleValue() / v.doubleValue(),result.imaginaryComponent().doubleValue() / v.doubleValue());

        }
        return result;
    }

    @Override
    public IComplexNumber divi(Number v) {
        return divi(v,this);
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
    public String toString() {
        return super.toString();
    }



    @Override
    public Double realComponent() {
        return real;
    }

    @Override
    public Double imaginaryComponent() {
        return  imag;
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
        return Math.acos(realComponent()/ absoluteValue());
    }

    @Override
    public IComplexDouble invi() {
        double d = realComponent() * realComponent() + imaginaryComponent() * imaginaryComponent();
        set(realComponent() / d,-imaginaryComponent() / d);
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
    public IComplexDouble negi() {
        set(-realComponent(),-imaginaryComponent());
        return this;
    }

    @Override
    public IComplexDouble sqrt() {
        double a = absoluteValue();
        double s2 = Math.sqrt(2);
        double p =  Math.sqrt(a + realComponent())/s2;
        double q = Math.sqrt(a - realComponent())/s2 * Math.signum(imaginaryComponent());
        return Nd4j.createDouble(p, q);
    }

    /**
     * Comparing two DoubleComplex values.
     *
     * @param o
     */
    @Override
    public boolean equals(Object o) {
        return super.equals(o);
    }



}
