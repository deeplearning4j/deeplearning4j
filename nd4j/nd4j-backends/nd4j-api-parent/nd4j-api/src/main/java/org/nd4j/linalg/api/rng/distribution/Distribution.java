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

package org.nd4j.linalg.api.rng.distribution;

import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A probability distribution
 *
 * @author Adam Gibson
 */
public interface Distribution {

    /**
     * For a random variable {@code X} whose values are distributed according
     * to this distribution, this method returns {@code P(X = x)}. In other
     * words, this method represents the probability mass function (PMF)
     * for the distribution.
     *
     * @param x the point at which the PMF is evaluated
     * @return the value of the probability mass function at point {@code x}
     */
    double probability(double x);

    /**
     * Returns the probability density function (PDF) of this distribution
     * evaluated at the specified point {@code x}. In general, the PDF is
     * the derivative of the {@link #cumulativeProbability(double) CDF}.
     * If the derivative does not exist at {@code x}, then an appropriate
     * replacement should be returned, e.g. {@code Double.POSITIVE_INFINITY},
     * {@code Double.NaN}, or  the limit inferior or limit superior of the
     * difference quotient.
     *
     * @param x the point at which the PDF is evaluated
     * @return the value of the probability density function at point {@code x}
     */
    double density(double x);

    /**
     * For a random variable {@code X} whose values are distributed according
     * to this distribution, this method returns {@code P(X <= x)}. In other
     * words, this method represents the (cumulative) distribution function
     * (CDF) for this distribution.
     *
     * @param x the point at which the CDF is evaluated
     * @return the probability that a random variable with this
     * distribution takes a value less than or equal to {@code x}
     */
    double cumulativeProbability(double x);

    /**
     * For a random variable {@code X} whose values are distributed according
     * to this distribution, this method returns {@code P(x0 < X <= x1)}.
     *
     * @param x0 the exclusive lower bound
     * @param x1 the inclusive upper bound
     * @return the probability that a random variable with this distribution
     * takes a value between {@code x0} and {@code x1},
     * excluding the lower and including the upper endpoint
     * @throws org.apache.commons.math3.exception.NumberIsTooLargeException if {@code x0 > x1}
     * @deprecated As of 3.1. In 4.0, this method will be renamed
     * {@code probability(double x0, double x1)}.
     */
    @Deprecated
    double cumulativeProbability(double x0, double x1) throws NumberIsTooLargeException;

    /**
     * Computes the quantile function of this distribution. For a random
     * variable {@code X} distributed according to this distribution, the
     * returned value is
     * <ul>
     * <li><code>inf{x in R | P(X<=x) >= p}</code> for {@code 0 < p <= 1},</li>
     * <li><code>inf{x in R | P(X<=x) > 0}</code> for {@code p = 0}.</li>
     * </ul>
     *
     * @param p the cumulative probability
     * @return the smallest {@code p}-quantile of this distribution
     * (largest 0-quantile for {@code p = 0})
     * @throws org.apache.commons.math3.exception.OutOfRangeException if {@code p < 0} or {@code p > 1}
     */
    double inverseCumulativeProbability(double p) throws OutOfRangeException;

    /**
     * Use this method to get the numerical value of the mean of this
     * distribution.
     *
     * @return the mean or {@code Double.NaN} if it is not defined
     */
    double getNumericalMean();

    /**
     * Use this method to get the numerical value of the variance of this
     * distribution.
     *
     * @return the variance (possibly {@code Double.POSITIVE_INFINITY} as
     * for certain cases in {@link org.apache.commons.math3.distribution.TDistribution}) or {@code Double.NaN} if it
     * is not defined
     */
    double getNumericalVariance();

    /**
     * Access the lower bound of the support. This method must return the same
     * value as {@code inverseCumulativeProbability(0)}. In other words, this
     * method must return
     * <p><code>inf {x in R | P(X <= x) > 0}</code>.</p>
     *
     * @return lower bound of the support (might be
     * {@code Double.NEGATIVE_INFINITY})
     */
    double getSupportLowerBound();

    /**
     * Access the upper bound of the support. This method must return the same
     * value as {@code inverseCumulativeProbability(1)}. In other words, this
     * method must return
     * <p><code>inf {x in R | P(X <= x) = 1}</code>.</p>
     *
     * @return upper bound of the support (might be
     * {@code Double.POSITIVE_INFINITY})
     */
    double getSupportUpperBound();

    /**
     * Whether or not the lower bound of support is in the domain of the density
     * function.  Returns true iff {@code getSupporLowerBound()} is finite and
     * {@code density(getSupportLowerBound())} returns a non-NaN, non-infinite
     * value.
     *
     * @return true if the lower bound of support is finite and the density
     * function returns a non-NaN, non-infinite value there
     * @deprecated to be removed in 4.0
     */
    boolean isSupportLowerBoundInclusive();

    /**
     * Whether or not the upper bound of support is in the domain of the density
     * function.  Returns true iff {@code getSupportUpperBound()} is finite and
     * {@code density(getSupportUpperBound())} returns a non-NaN, non-infinite
     * value.
     *
     * @return true if the upper bound of support is finite and the density
     * function returns a non-NaN, non-infinite value there
     * @deprecated to be removed in 4.0
     */
    boolean isSupportUpperBoundInclusive();

    /**
     * Use this method to get information about whether the support is connected,
     * i.e. whether all values between the lower and upper bound of the support
     * are included in the support.
     *
     * @return whether the support is connected or not
     */
    boolean isSupportConnected();

    /**
     * Reseed the random generator used to generate samples.
     *
     * @param seed the new seed
     */
    void reseedRandomGenerator(long seed);

    /**
     * Generate a random value sampled from this distribution.
     *
     * @return a random value.
     */
    double sample();

    /**
     * Generate a random sample from the distribution.
     *
     * @param sampleSize the number of random values to generate
     * @return an array representing the random sample
     * @throws org.apache.commons.math3.exception.NotStrictlyPositiveException if {@code sampleSize} is not positive
     */
    double[] sample(long sampleSize);

    /**
     * Sample the given shape
     *
     * @param shape the given shape
     * @return an ndarray with random samples
     * from this distribution
     */
    INDArray sample(int[] shape);

    INDArray sample(long[] shape);


    /**
     * Fill the target array by sampling from the distribution
     *
     * @param target  target array
     * @return an ndarray with random samples from this distribution
     */
    INDArray sample(INDArray target);

}
