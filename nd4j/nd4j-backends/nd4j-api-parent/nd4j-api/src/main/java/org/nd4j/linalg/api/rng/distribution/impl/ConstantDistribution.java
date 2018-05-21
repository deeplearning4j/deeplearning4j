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

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.BaseDistribution;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * This is not real distribution. It'll generate valueOf array with requested shape.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class ConstantDistribution extends BaseDistribution {
    /**
     * Default inverse cumulative probability accuracy.
     *
     * @since 2.1
     */
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    /**
     * Serializable version identifier.
     */
    private static final long serialVersionUID = 8589540077390120676L;

    /**
     * Mean of this distribution.
     */
    private double value;

    public ConstantDistribution(double value) {
        this.value = value;
        this.random = Nd4j.getRandom();
    }
/*
    max doesn't want this distripution
    public OrthogonalDistribution(@NonNull INDArray gains) {
        this.gains = gains;
        this.random = Nd4j.getRandom();
    }
*/
    /**
     * Access the mean.
     *
     * @return the mean for this distribution.
     */
    public double getMean() {
        throw new UnsupportedOperationException();
    }

    /**
     * Access the standard deviation.
     *
     * @return the standard deviation for this distribution.
     */
    public double getStandardDeviation() {
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc}
     */
    public double density(double x) {
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc}
     * <p/>
     * If {@code x} is more than 40 standard deviations from the mean, 0 or 1
     * is returned, as in these cases the actual value is within
     * {@code Double.MIN_VALUE} of 0 or 1.
     */
    public double cumulativeProbability(double x) {
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc}
     *
     * @since 3.2
     */
    @Override
    public double inverseCumulativeProbability(final double p) throws OutOfRangeException {
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc}
     *
     * @deprecated See {@link org.apache.commons.math3.distribution.RealDistribution#cumulativeProbability(double, double)}
     */
    @Override
    @Deprecated
    public double cumulativeProbability(double x0, double x1) throws NumberIsTooLargeException {
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double probability(double x0, double x1) throws NumberIsTooLargeException {
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected double getSolverAbsoluteAccuracy() {
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc}
     * <p/>
     * For mean parameter {@code mu}, the mean is {@code mu}.
     */
    public double getNumericalMean() {
        return getMean();
    }

    /**
     * {@inheritDoc}
     * <p/>
     * For standard deviation parameter {@code s}, the variance is {@code s^2}.
     */
    public double getNumericalVariance() {
        final double s = getStandardDeviation();
        return s * s;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The lower bound of the support is always negative infinity
     * no matter the parameters.
     *
     * @return lower bound of the support (always
     * {@code Double.NEGATIVE_INFINITY})
     */
    public double getSupportLowerBound() {
        return Double.NEGATIVE_INFINITY;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The upper bound of the support is always positive infinity
     * no matter the parameters.
     *
     * @return upper bound of the support (always
     * {@code Double.POSITIVE_INFINITY})
     */
    public double getSupportUpperBound() {
        return Double.POSITIVE_INFINITY;
    }

    /**
     * {@inheritDoc}
     */
    public boolean isSupportLowerBoundInclusive() {
        return false;
    }

    /**
     * {@inheritDoc}
     */
    public boolean isSupportUpperBoundInclusive() {
        return false;
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
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray sample(int[] shape) {
        return sample(Nd4j.createUninitialized(shape));
    }


    @Override
    public INDArray sample(INDArray target) {
        return target.assign(value);
    }
}
