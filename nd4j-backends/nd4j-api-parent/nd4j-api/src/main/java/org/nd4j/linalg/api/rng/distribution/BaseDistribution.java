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

package org.nd4j.linalg.api.rng.distribution;

import java.util.Iterator;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.solvers.UnivariateSolverUtils;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Base distribution derived from apache commons math
 * http://commons.apache.org/proper/commons-math/
 * <p/>
 * (specifically the {@link org.apache.commons.math3.distribution.AbstractRealDistribution}
 *
 * @author Adam Gibson
 */
public abstract class BaseDistribution implements Distribution {
    protected Random random;
    protected double solverAbsoluteAccuracy;


    public BaseDistribution(Random rng) {
        this.random = rng;
    }


    public BaseDistribution() {
        this(Nd4j.getRandom());
    }

    /**
     * For a random variable {@code X} whose values are distributed according
     * to this distribution, this method returns {@code P(x0 < X <= x1)}.
     *
     * @param x0 Lower bound (excluded).
     * @param x1 Upper bound (included).
     * @return the probability that a random variable with this distribution
     * takes a value between {@code x0} and {@code x1}, excluding the lower
     * and including the upper endpoint.
     * @throws org.apache.commons.math3.exception.NumberIsTooLargeException if {@code x0 > x1}.
     *                                                                      <p/>
     *                                                                      The default implementation uses the identity
     *                                                                      {@code P(x0 < X <= x1) = P(X <= x1) - P(X <= x0)}
     * @since 3.1
     */

    public double probability(double x0,
                              double x1) {
        if (x0 > x1) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_ENDPOINT_ABOVE_UPPER_ENDPOINT,
                    x0, x1, true);
        }
        return cumulativeProbability(x1) - cumulativeProbability(x0);
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The default implementation returns
     * <ul>
     * <li>{@link #getSupportLowerBound()} for {@code p = 0},</li>
     * <li>{@link #getSupportUpperBound()} for {@code p = 1}.</li>
     * </ul>
     */
    @Override
    public double inverseCumulativeProbability(final double p) throws OutOfRangeException {
        /*
         * IMPLEMENTATION NOTES
         * --------------------
         * Where applicable, use is made of the one-sided Chebyshev inequality
         * to bracket the root. This inequality states that
         * P(X - mu >= k * sig) <= 1 / (1 + k^2),
         * mu: mean, sig: standard deviation. Equivalently
         * 1 - P(X < mu + k * sig) <= 1 / (1 + k^2),
         * F(mu + k * sig) >= k^2 / (1 + k^2).
         *
         * For k = sqrt(p / (1 - p)), we find
         * F(mu + k * sig) >= p,
         * and (mu + k * sig) is an upper-bound for the root.
         *
         * Then, introducing Y = -X, mean(Y) = -mu, sd(Y) = sig, and
         * P(Y >= -mu + k * sig) <= 1 / (1 + k^2),
         * P(-X >= -mu + k * sig) <= 1 / (1 + k^2),
         * P(X <= mu - k * sig) <= 1 / (1 + k^2),
         * F(mu - k * sig) <= 1 / (1 + k^2).
         *
         * For k = sqrt((1 - p) / p), we find
         * F(mu - k * sig) <= p,
         * and (mu - k * sig) is a lower-bound for the root.
         *
         * In cases where the Chebyshev inequality does not apply, geometric
         * progressions 1, 2, 4, ... and -1, -2, -4, ... are used to bracket
         * the root.
         */
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }

        double lowerBound = getSupportLowerBound();
        if (p == 0.0) {
            return lowerBound;
        }

        double upperBound = getSupportUpperBound();
        if (p == 1.0) {
            return upperBound;
        }

        final double mu = getNumericalMean();
        final double sig = FastMath.sqrt(getNumericalVariance());
        final boolean chebyshevApplies;
        chebyshevApplies = !(Double.isInfinite(mu) || Double.isNaN(mu) ||
                Double.isInfinite(sig) || Double.isNaN(sig));

        if (lowerBound == Double.NEGATIVE_INFINITY) {
            if (chebyshevApplies) {
                lowerBound = mu - sig * FastMath.sqrt((1. - p) / p);
            } else {
                lowerBound = -1.0;
                while (cumulativeProbability(lowerBound) >= p) {
                    lowerBound *= 2.0;
                }
            }
        }

        if (upperBound == Double.POSITIVE_INFINITY) {
            if (chebyshevApplies) {
                upperBound = mu + sig * FastMath.sqrt(p / (1. - p));
            } else {
                upperBound = 1.0;
                while (cumulativeProbability(upperBound) < p) {
                    upperBound *= 2.0;
                }
            }
        }

        final UnivariateFunction toSolve = new UnivariateFunction() {

            public double value(final double x) {
                return cumulativeProbability(x) - p;
            }
        };

        double x = UnivariateSolverUtils.solve(toSolve,
                lowerBound,
                upperBound,
                getSolverAbsoluteAccuracy());

        if (!isSupportConnected()) {
            /* Test for plateau. */
            final double dx = getSolverAbsoluteAccuracy();
            if (x - dx >= getSupportLowerBound()) {
                double px = cumulativeProbability(x);
                if (cumulativeProbability(x - dx) == px) {
                    upperBound = x;
                    while (upperBound - lowerBound > dx) {
                        final double midPoint = 0.5 * (lowerBound + upperBound);
                        if (cumulativeProbability(midPoint) < px) {
                            lowerBound = midPoint;
                        } else {
                            upperBound = midPoint;
                        }
                    }
                    return upperBound;
                }
            }
        }
        return x;
    }

    /**
     * Returns the solver absolute accuracy for inverse cumulative computation.
     * You can override this method in order to use a Brent solver with an
     * absolute accuracy different from the default.
     *
     * @return the maximum absolute error in inverse cumulative probability estimates
     */
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void reseedRandomGenerator(long seed) {
        random.setSeed(seed);
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The default implementation uses the
     * <a href="http://en.wikipedia.org/wiki/Inverse_transform_sampling">
     * inversion method.
     * </a>
     */
    @Override
    public double sample() {
        return inverseCumulativeProbability(random.nextDouble());
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The default implementation generates the sample by calling
     * {@link #sample()} in a loop.
     */
    @Override
    public double[] sample(int sampleSize) {
        if (sampleSize <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_SAMPLES,
                    sampleSize);
        }
        double[] out = new double[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            out[i] = sample();
        }
        return out;
    }

    /**
     * {@inheritDoc}
     *
     * @return zero.
     * @since 3.1
     */
    @Override
    public double probability(double x) {
        return 0d;
    }

    @Override
    public INDArray sample(int[] shape) {
        INDArray ret = Nd4j.create(shape);
        Iterator<int[]> idxIter = new NdIndexIterator(shape);	//For consistent values irrespective of c vs. fortran ordering
        int len = ret.length();
        for( int i=0; i<len; i++ ){
        	ret.putScalar(idxIter.next(), sample());
        }
        return ret;
    }
}
