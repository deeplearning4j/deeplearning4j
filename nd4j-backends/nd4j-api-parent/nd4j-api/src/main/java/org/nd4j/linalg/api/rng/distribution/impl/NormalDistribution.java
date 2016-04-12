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

package org.nd4j.linalg.api.rng.distribution.impl;

import java.util.Iterator;

import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.rng.distribution.BaseDistribution;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Base distribution derived from apache commons math
 * http://commons.apache.org/proper/commons-math/
 * <p/>
 * (specifically the {@link org.apache.commons.math3.distribution.NormalDistribution}
 *
 * @author Adam Gibson
 */
public class NormalDistribution extends BaseDistribution {
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
     * &radic;(2 &pi;)
     */
    private static final double SQRT2PI = FastMath.sqrt(2 * FastMath.PI);
    /**
     * &radic;(2)
     */
    private static final double SQRT2 = FastMath.sqrt(2.0);
    /**
     * Standard deviation of this distribution.
     */
    private final double standardDeviation;
    /**
     * Mean of this distribution.
     */
    private double mean;
    private INDArray means;
    /**
     * Inverse cumulative probability accuracy.
     */
    private double solverAbsoluteAccuracy;

    public NormalDistribution(Random rng, double standardDeviation, INDArray means) {
        super(rng);
        this.standardDeviation = standardDeviation;
        this.means = means;
    }

    public NormalDistribution(double standardDeviation, INDArray means) {
        this.standardDeviation = standardDeviation;
        this.means = means;
    }

    /**
     * Create a normal distribution with mean equal to zero and standard
     * deviation equal to one.
     */
    public NormalDistribution() {
        this(0, 1);
    }

    /**
     * Create a normal distribution using the given mean and standard deviation.
     *
     * @param mean Mean for this distribution.
     * @param sd   Standard deviation for this distribution.
     * @throws org.apache.commons.math3.exception.NotStrictlyPositiveException if {@code sd <= 0}.
     */
    public NormalDistribution(double mean, double sd)
            throws NotStrictlyPositiveException {
        this(mean, sd, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    /**
     * Create a normal distribution using the given mean, standard deviation and
     * inverse cumulative distribution accuracy.
     *
     * @param mean               Mean for this distribution.
     * @param sd                 Standard deviation for this distribution.
     * @param inverseCumAccuracy Inverse cumulative probability accuracy.
     * @throws NotStrictlyPositiveException if {@code sd <= 0}.
     * @since 2.1
     */
    public NormalDistribution(double mean, double sd, double inverseCumAccuracy)
            throws NotStrictlyPositiveException {
        this(Nd4j.getRandom(), mean, sd, inverseCumAccuracy);
    }

    /**
     * Creates a normal distribution.
     *
     * @param rng                Random number generator.
     * @param mean               Mean for this distribution.
     * @param sd                 Standard deviation for this distribution.
     * @param inverseCumAccuracy Inverse cumulative probability accuracy.
     * @throws NotStrictlyPositiveException if {@code sd <= 0}.
     * @since 3.1
     */
    public NormalDistribution(Random rng,
                              double mean,
                              double sd,
                              double inverseCumAccuracy)
            throws NotStrictlyPositiveException {
        super(rng);

        if (sd <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.STANDARD_DEVIATION, sd);
        }

        this.mean = mean;
        standardDeviation = sd;
        solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    public NormalDistribution(INDArray mean, double std) {
        this.means = mean;
        this.standardDeviation = std;
        this.random = Nd4j.getRandom();
    }

    /**
     * Access the mean.
     *
     * @return the mean for this distribution.
     */
    public double getMean() {
        return mean;
    }

    /**
     * Access the standard deviation.
     *
     * @return the standard deviation for this distribution.
     */
    public double getStandardDeviation() {
        return standardDeviation;
    }

    /**
     * {@inheritDoc}
     */
    public double density(double x) {
        if (means != null)
            throw new IllegalStateException("Unable to sample from more than one mean");
        final double x0 = x - mean;
        final double x1 = x0 / standardDeviation;
        return FastMath.exp(-0.5 * x1 * x1) / (standardDeviation * SQRT2PI);
    }

    /**
     * {@inheritDoc}
     * <p/>
     * If {@code x} is more than 40 standard deviations from the mean, 0 or 1
     * is returned, as in these cases the actual value is within
     * {@code Double.MIN_VALUE} of 0 or 1.
     */
    public double cumulativeProbability(double x) {
        if (means != null)
            throw new IllegalStateException("Unable to sample from more than one mean");
        final double dev = x - mean;
        if (FastMath.abs(dev) > 40 * standardDeviation) {
            return dev < 0 ? 0.0d : 1.0d;
        }
        return 0.5 * (1 + Erf.erf(dev / (standardDeviation * SQRT2)));
    }

    /**
     * {@inheritDoc}
     *
     * @since 3.2
     */
    @Override
    public double inverseCumulativeProbability(final double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }
        if (means != null)
            throw new IllegalStateException("Unable to sample from more than one mean");

        return mean + standardDeviation * SQRT2 * Erf.erfInv(2 * p - 1);
    }

    /**
     * {@inheritDoc}
     *
     * @deprecated See {@link org.apache.commons.math3.distribution.RealDistribution#cumulativeProbability(double, double)}
     */
    @Override
    @Deprecated
    public double cumulativeProbability(double x0, double x1)
            throws NumberIsTooLargeException {
        return probability(x0, x1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double probability(double x0,
                              double x1)
            throws NumberIsTooLargeException {
        if (x0 > x1) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_ENDPOINT_ABOVE_UPPER_ENDPOINT,
                    x0, x1, true);
        }
        final double denom = standardDeviation * SQRT2;
        final double v0 = (x0 - mean) / denom;
        final double v1 = (x1 - mean) / denom;
        return 0.5 * Erf.erf(v0, v1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
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
        if (means != null)
            throw new IllegalStateException("Unable to sample from more than one mean");
        return standardDeviation * random.nextGaussian() + mean;
    }

    @Override
    public INDArray sample(int[] shape) {
        INDArray ret = Nd4j.create(shape);
        Iterator<int[]> idxIter = new NdIndexIterator(shape);	//For consistent values irrespective of c vs. fortran ordering
        int len = ret.length();
        if( means != null ){
        	for( int i=0; i<len; i++ ){
        		int[] idx = idxIter.next();
        		ret.putScalar(idx, standardDeviation * random.nextGaussian() + means.getDouble(idx));
        	}
        } else {
        	for( int i=0; i<len; i++ ){
        		ret.putScalar(idxIter.next(), standardDeviation * random.nextGaussian() + mean);
        	}
        }
        return ret;
    }
}
