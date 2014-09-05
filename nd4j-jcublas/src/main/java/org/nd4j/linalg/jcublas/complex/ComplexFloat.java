package org.nd4j.linalg.jcublas.complex;

import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.NDArrays;

/**
 * Complex float
 * @author Adam Gibson
 */
public class ComplexFloat implements IComplexFloat {


    public final static ComplexFloat UNIT = new ComplexFloat(1,0);
    public final static ComplexFloat NEG = new ComplexFloat(-1,0);
    public final static ComplexFloat ZERO = new ComplexFloat(0,0);
    private float real,imag;

    public ComplexFloat(float real, float imag) {
        this.real = real;
        this.imag = imag;

    }

    public ComplexFloat(float real) {
        this(real,0);
    }



    /**
     * Convert to a double
     *
     * @return this complex number as a double
     */
    @Override
    public IComplexDouble asDouble() {
        return NDArrays.createDouble(realComponent(),imaginaryComponent());
    }

    /**
     * Convert to a float
     *
     * @return this complex number as a float
     */
    @Override
    public ComplexFloat asFloat() {
        return this;
    }

    @Override
    public ComplexFloat dup() {
        return new ComplexFloat(realComponent(), imaginaryComponent());
    }


    @Override
    public ComplexFloat conji() {
       set(realComponent(), -imaginaryComponent());
        return this;
    }

    @Override
    public ComplexFloat conj() {
        return dup().conji();
    }

    @Override
    public IComplexNumber set(Number real, Number imag) {
        set( real.floatValue(),imag.floatValue());
        return this;
    }

    @Override
    public IComplexNumber copy(IComplexNumber other) {

        return NDArrays.createFloat(other.realComponent().floatValue(),other.imaginaryComponent().floatValue());

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
            set(realComponent() + c.realComponent().floatValue(),imaginaryComponent() + result.imaginaryComponent().floatValue());
        } else {
            result.set(result.realComponent().floatValue() + c.realComponent().floatValue(),result.imaginaryComponent().floatValue() + c.imaginaryComponent().floatValue());

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
            set(realComponent()+ a.floatValue(),imaginaryComponent());
        } else {
            result.set(result.realComponent().floatValue() + a.floatValue(),imaginaryComponent());

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
            set(realComponent()- c.realComponent().floatValue(),imaginaryComponent() - result.imaginaryComponent().floatValue());
        } else {
            result.set(result.realComponent().floatValue() - c.realComponent().floatValue(),result.imaginaryComponent().floatValue() - c.imaginaryComponent().floatValue());

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
            set(realComponent()- a.floatValue(),imaginaryComponent());
        } else {
            result.set(result.realComponent().floatValue() - a.floatValue(),imaginaryComponent());

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
        float newR = realComponent()* c.realComponent().floatValue() - imaginaryComponent() * c.imaginaryComponent().floatValue();
        float newI = realComponent()* c.imaginaryComponent().floatValue() + imaginaryComponent() * c.realComponent().floatValue();
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
            set(realComponent()+ v.floatValue(),imaginaryComponent());
        } else {
            result.set(result.realComponent().floatValue() + v.floatValue(),imaginaryComponent());

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
        float d = c.realComponent().floatValue() * c.realComponent().floatValue() + c.imaginaryComponent().floatValue() * c.imaginaryComponent().floatValue();
        float newR = (realComponent()* c.realComponent().floatValue() + imaginaryComponent() * c.imaginaryComponent().floatValue()) / d;
        float newI = (imaginaryComponent() * c.realComponent().floatValue() - realComponent()* c.imaginaryComponent().floatValue()) / d;
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
            set(realComponent()/ v.floatValue(),imaginaryComponent());
        } else {
            result.set(result.realComponent().floatValue() / v.floatValue(),imaginaryComponent());

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
        return this;
    }

    @Override
    public IComplexFloat div(float v) {
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
        return (float) Math.acos(realComponent()/ absoluteValue());
    }

    @Override
    public ComplexFloat invi() {
        float d = realComponent() * realComponent() + imaginaryComponent() * imaginaryComponent();
        set(realComponent() / d,-imaginaryComponent() / d);
        return this;
    }

    @Override
    public ComplexFloat inv() {
        return dup().invi();
    }

    @Override
    public ComplexFloat neg() {
        return dup().negi();
    }

    @Override
    public ComplexFloat negi() {
        set(-realComponent(),-imaginaryComponent());
        return this;
    }

    @Override
    public ComplexFloat sqrt() {
        float a = absoluteValue();
        float s2 = (float)Math.sqrt(2);
        float p = (float)Math.sqrt(a + realComponent())/s2;
        float q = (float)Math.sqrt(a - realComponent())/s2 * Math.signum(imaginaryComponent());
        return new ComplexFloat(p, q);
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


}
