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

/**
 * Baseline interface for a complex number with realComponent and imaginary components.
 * <p/>
 * Based off of the jblas api by mikio braun
 *
 * @author Adam Gibson
 */
public interface IComplexNumber {

    /**
     * Set the real and imaginary components
     *
     * @param real the real numbers
     * @param imag the imaginary components
     * @return the imaginary components
     */
    public IComplexNumber set(Number real, Number imag);

    /**
     * The real component of this number
     *
     * @return the real component of this number
     */
    public Number realComponent();

    /**
     * The imaginary component of this number
     *
     * @return the real component of this number
     */
    public Number imaginaryComponent();

    /**
     * Clone
     *
     * @return
     */
    public IComplexNumber dup();

    public IComplexNumber copy(IComplexNumber other);

    /**
     * Add two complex numbers in-place
     */
    public IComplexNumber addi(IComplexNumber c, IComplexNumber result);

    /**
     * Add two complex numbers in-place storing the result in this.
     */
    public IComplexNumber addi(IComplexNumber c);

    /**
     * Add two complex numbers.
     */
    public IComplexNumber add(IComplexNumber c);

    /**
     * Add a realComponent number to a complex number in-place.
     */
    public IComplexNumber addi(Number a, IComplexNumber result);

    /**
     * Add a realComponent number to complex number in-place, storing the result in this.
     */
    public IComplexNumber addi(Number c);

    /**
     * Add a realComponent number to a complex number.
     */
    public IComplexNumber add(Number c);

    /**
     * Subtract two complex numbers, in-place
     */
    public IComplexNumber subi(IComplexNumber c, IComplexNumber result);

    public IComplexNumber subi(IComplexNumber c);

    /**
     * Subtract two complex numbers
     */
    public IComplexNumber sub(IComplexNumber c);

    public IComplexNumber subi(Number a, IComplexNumber result);

    public IComplexNumber subi(Number a);

    public IComplexNumber sub(Number r);


    /**
     * Subtract two complex numbers
     */
    public IComplexNumber rsub(IComplexNumber c);

    public IComplexNumber rsubi(Number a, IComplexNumber result);

    public IComplexNumber rsubi(Number a);

    public IComplexNumber rsub(Number r);

    /**
     * Multiply two complex numbers, inplace
     */
    public IComplexNumber muli(IComplexNumber c, IComplexNumber result);

    public IComplexNumber muli(IComplexNumber c);

    /**
     * Multiply two complex numbers
     */
    public IComplexNumber mul(IComplexNumber c);

    public IComplexNumber mul(Number v);

    public IComplexNumber muli(Number v, IComplexNumber result);

    public IComplexNumber muli(Number v);

    /**
     * Divide two complex numbers
     */
    public IComplexNumber div(IComplexNumber c);

    /**
     * Divide two complex numbers, in-place
     */
    public IComplexNumber divi(IComplexNumber c, IComplexNumber result);

    public IComplexNumber divi(IComplexNumber c);

    public IComplexNumber divi(Number v, IComplexNumber result);

    public IComplexNumber divi(Number v);

    public IComplexNumber div(Number v);


    /**
     * Divide two complex numbers
     */
    public IComplexNumber rdiv(IComplexNumber c);

    /**
     * Divide two complex numbers, in-place
     */
    public IComplexNumber rdivi(IComplexNumber c, IComplexNumber result);

    public IComplexNumber rdivi(IComplexNumber c);

    public IComplexNumber rdivi(Number v, IComplexNumber result);

    public IComplexNumber rdivi(Number v);

    public IComplexNumber rdiv(Number v);

    /**
     * Return the absolute value
     */
    public Number absoluteValue();

    /**
     * Returns the argument of a complex number.
     */
    public Number complexArgument();

    public IComplexNumber invi();

    public IComplexNumber inv();

    /**
     * The negation of this complex number
     *
     * @return
     */
    public IComplexNumber neg();

    /**
     * The inplace negation of this number
     *
     * @return
     */
    public IComplexNumber negi();

    /**
     * The inplace conjugate of this
     * number
     *
     * @return
     */
    public IComplexNumber conji();

    /**
     * The  conjugate of this
     * number
     *
     * @return
     */
    public IComplexNumber conj();

    /**
     * The  sqrt of this
     * number
     *
     * @return
     */
    public IComplexNumber sqrt();


    public boolean eq(IComplexNumber c);

    public boolean ne(IComplexNumber c);

    /**
     * Whether this number is
     * wholly zero or not
     *
     * @return true if the number is wholly
     * zero false otherwise
     */
    public boolean isZero();

    /**
     * Returns whether the number
     * only has a real component (0 for imaginary)
     *
     * @return true if the number has only a real component or not
     */
    public boolean isReal();

    /**
     * Returns whether the number
     * only has a imaginary component (0 for real)
     *
     * @return true if the number has only a real component or not
     */
    public boolean isImag();

    /**
     * Convert to a float
     *
     * @return this complex number as a float
     */
    public IComplexFloat asFloat();

    /**
     * Convert to a double
     *
     * @return this complex number as a double
     */
    public IComplexDouble asDouble();

    /**
     * Equals returning a complex number
     *
     * @param num the number to compare
     * @return 1 if equal 0 otherwise
     */
    public IComplexNumber eqc(IComplexNumber num);

    /**
     * Not Equals returning a complex number
     *
     * @param num the number to compare
     * @return 1 if not equal 0 otherwise
     */
    public IComplexNumber neqc(IComplexNumber num);

    /**
     * Greater than returning a complex number
     *
     * @param num the number to compare
     * @return 1 if greater than 0 otherwise
     */
    public IComplexNumber gt(IComplexNumber num);

    /**
     * Less than returning a complex number
     *
     * @param num the number to compare
     * @return 1 if less than 0 otherwise
     */
    public IComplexNumber lt(IComplexNumber num);

    /**
     * Reverse subtract a number
     *
     * @param c the complex number to reverse subtract
     * @return the reverse subtracted number
     */
    IComplexNumber rsubi(IComplexNumber c);

    /**
     * Set a complex number's components to be this ones
     *
     * @param set the complex number to set
     * @return a reference to this
     */
    IComplexNumber set(IComplexNumber set);

    /**
     * Reverse subtraction
     *
     * @param a      the number to subtract
     * @param result the result to set
     * @return the result
     */
    IComplexNumber rsubi(IComplexNumber a, IComplexNumber result);

}
