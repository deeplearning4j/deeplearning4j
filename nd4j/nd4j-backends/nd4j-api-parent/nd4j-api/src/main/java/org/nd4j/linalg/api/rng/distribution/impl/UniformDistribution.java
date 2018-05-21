/*-
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

package org.nd4j.linalg.api.rng.distribution.impl;

import lombok.val;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.BaseDistribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Iterator;

/**
 * Base distribution derived from apache commons math
 * http://commons.apache.org/proper/commons-math/
 * <p/>
 * (specifically the {@link org.apache.commons.math3.distribution.UniformIntegerDistribution}
 *
 * @author Adam Gibson
 */
public class UniformDistribution extends BaseDistribution {
    private double upper, lower;

    /**
     * Create a uniform real distribution using the given lower and upper
     * bounds.
     *
     * @param lower Lower bound of this distribution (inclusive).
     * @param upper Upper bound of this distribution (exclusive).
     * @throws NumberIsTooLargeException if {@code lower >= upper}.
     */
    public UniformDistribution(double lower, double upper) throws NumberIsTooLargeException {
        this(Nd4j.getRandom(), lower, upper);
    }


    /**
     * Creates a uniform distribution.
     *
     * @param rng   Random number generator.
     * @param lower Lower bound of this distribution (inclusive).
     * @param upper Upper bound of this distribution (exclusive).
     * @throws NumberIsTooLargeException if {@code lower >= upper}.
     * @since 3.1
     */
    public UniformDistribution(org.nd4j.linalg.api.rng.Random rng, double lower, double upper)
                    throws NumberIsTooLargeException {
        super(rng);
        if (lower >= upper) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND, lower, upper,
                            false);
        }

        this.lower = lower;
        this.upper = upper;
    }

    /**
     * {@inheritDoc}
     */
    public double density(double x) {
        if (x < lower || x > upper) {
            return 0.0;
        }
        return 1 / (upper - lower);
    }

    /**
     * {@inheritDoc}
     */
    public double cumulativeProbability(double x) {
        if (x <= lower) {
            return 0;
        }
        if (x >= upper) {
            return 1;
        }
        return (x - lower) / (upper - lower);
    }

    @Override
    public double cumulativeProbability(double x0, double x1) throws NumberIsTooLargeException {
        return 0;
    }

    @Override
    public double inverseCumulativeProbability(final double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }
        return p * (upper - lower) + lower;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * For lower bound {@code lower} and upper bound {@code upper}, the mean is
     * {@code 0.5 * (lower + upper)}.
     */
    public double getNumericalMean() {
        return 0.5 * (lower + upper);
    }

    /**
     * {@inheritDoc}
     * <p/>
     * For lower bound {@code lower} and upper bound {@code upper}, the
     * variance is {@code (upper - lower)^2 / 12}.
     */
    public double getNumericalVariance() {
        double ul = upper - lower;
        return ul * ul / 12;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The lower bound of the support is equal to the lower bound parameter
     * of the distribution.
     *
     * @return lower bound of the support
     */
    public double getSupportLowerBound() {
        return lower;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The upper bound of the support is equal to the upper bound parameter
     * of the distribution.
     *
     * @return upper bound of the support
     */
    public double getSupportUpperBound() {
        return upper;
    }

    /**
     * {@inheritDoc}
     */
    public boolean isSupportLowerBoundInclusive() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    public boolean isSupportUpperBoundInclusive() {
        return true;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The support of this distribution is connected.
     *
     * @return {@code true}
     */
    public boolean isSupportConnected() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double sample() {
        final double u = random.nextDouble();
        return u * upper + (1 - u) * lower;
    }


    @Override
    public INDArray sample(int[] shape) {
        final INDArray ret = Nd4j.createUninitialized(shape, Nd4j.order());
        return sample(ret);
    }

    @Override
    public INDArray sample(INDArray ret) {
        if (random.getStatePointer() != null) {
            return Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.random.impl.UniformDistribution(
                    ret, lower, upper), random);
        } else {
            val idxIter = new NdIndexIterator(ret.shape()); //For consistent values irrespective of c vs. fortran ordering
            long len = ret.length();
            for (int i = 0; i < len; i++) {
                ret.putScalar(idxIter.next(), sample());
            }
            return ret;
        }
    }
}
