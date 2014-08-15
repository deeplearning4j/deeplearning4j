package org.deeplearning4j.linalg.api.complex;

/**
 * Baseline interface for a complex number with realComponent and imaginary components.
 *
 * Based off of the jblas api by mikio braun
 *
 * @author Adam Gibson
 */
public interface IComplexNumber {


    public IComplexNumber set(Number real, Number imag);

    public Number realComponent();

    public Number imaginaryComponent();
  
    public IComplexNumber dup();

    public IComplexNumber copy(IComplexNumber other);

    /** Add two complex numbers in-place */
    public IComplexNumber addi(IComplexNumber c, IComplexNumber result);

    /** Add two complex numbers in-place storing the result in this. */
    public IComplexNumber addi(IComplexNumber c);
    /** Add two complex numbers. */
    public IComplexNumber add(IComplexNumber c);

    /** Add a realComponent number to a complex number in-place. */
    public IComplexNumber addi(Number a, IComplexNumber result);

    /** Add a realComponent number to complex number in-place, storing the result in this. */
    public IComplexNumber addi(Number c);
    /** Add a realComponent number to a complex number. */
    public IComplexNumber add(Number c);
    
    /** Subtract two complex numbers, in-place */
    public IComplexNumber subi(IComplexNumber c, IComplexNumber result);

    public IComplexNumber subi(IComplexNumber c);

    /** Subtract two complex numbers */
    public IComplexNumber sub(IComplexNumber c);

    public IComplexNumber subi(Number a, IComplexNumber result);

    public IComplexNumber subi(Number a);

    public IComplexNumber sub(Number r);

    /** Multiply two complex numbers, inplace */
    public IComplexNumber muli(IComplexNumber c, IComplexNumber result);

    public IComplexNumber muli(IComplexNumber c);

    /** Multiply two complex numbers */
    public IComplexNumber mul(IComplexNumber c);

    public IComplexNumber mul(Number v);

    public IComplexNumber muli(Number v, IComplexNumber result);

    public IComplexNumber muli(Number v);

    /** Divide two complex numbers */
    public IComplexNumber div(IComplexNumber c);

    /** Divide two complex numbers, in-place */
    public IComplexNumber divi(IComplexNumber c, IComplexNumber result);

    public IComplexNumber divi(IComplexNumber c);
    public IComplexNumber divi(Number v, IComplexNumber result);

    public IComplexNumber divi(Number v);

    public IComplexNumber div(Number v);
    /** Return the absolute value */
    public Number absoluteValue();

    /** Returns the argument of a complex number. */
    public Number complexArgument();

    public IComplexNumber invi();

    public IComplexNumber inv();

    public IComplexNumber neg();

    public IComplexNumber negi();

    public IComplexNumber conji();

    public IComplexNumber conj();

    public IComplexNumber sqrt();


    public double arg();

    public boolean eq(IComplexNumber c);

    public boolean ne(IComplexNumber c);

    public boolean isZero();

    public boolean isReal();

    public boolean isImag();

    /**
     * Convert to a float
     * @return this complex number as a float
     */
    public IComplexFloat asFloat();
    /**
     * Convert to a double
     * @return this complex number as a double
     */
    public IComplexDouble asDouble();


}
