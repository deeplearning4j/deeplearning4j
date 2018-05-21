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

import org.apache.commons.math3.exception.NotPositiveException;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Beta;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.rng.distribution.BaseDistribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Iterator;

/**
 * Base distribution derived from apache commons math
 * http://commons.apache.org/proper/commons-math/
 * <p/>
 * (specifically the {@link org.apache.commons.math3.distribution.BinomialDistribution}
 *
 * @author Adam Gibson
 */
public class BinomialDistribution extends BaseDistribution {
    /**
     * The number of trials.
     */
    private final int numberOfTrials;
    /**
     * The probability of success.
     */
    private double probabilityOfSuccess;

    private INDArray p;

    /**
     * Create a binomial distribution with the given number of trials and
     * probability of success.
     *
     * @param trials Number of trials.
     * @param p      Probability of success.
     * @throws org.apache.commons.math3.exception.NotPositiveException if {@code trials < 0}.
     * @throws org.apache.commons.math3.exception.OutOfRangeException  if {@code p < 0} or {@code p > 1}.
     */
    public BinomialDistribution(int trials, double p) {
        this(Nd4j.getRandom(), trials, p);
    }

    /**
     * Creates a binomial distribution.
     *
     * @param rng    Random number generator.
     * @param trials Number of trials.
     * @param p      Probability of success.
     * @throws org.apache.commons.math3.exception.NotPositiveException if {@code trials < 0}.
     * @throws org.apache.commons.math3.exception.OutOfRangeException  if {@code p < 0} or {@code p > 1}.
     * @since 3.1
     */
    public BinomialDistribution(Random rng, int trials, double p) {
        super(rng);

        if (trials < 0) {
            throw new NotPositiveException(LocalizedFormats.NUMBER_OF_TRIALS, trials);
        }
        if (p < 0 || p > 1) {
            throw new OutOfRangeException(p, 0, 1);
        }

        probabilityOfSuccess = p;
        numberOfTrials = trials;
    }

    public BinomialDistribution(int n, INDArray p) {
        this.random = Nd4j.getRandom();
        this.numberOfTrials = n;
        this.p = p;
    }

    /**
     * Access the number of trials for this distribution.
     *
     * @return the number of trials.
     */
    public int getNumberOfTrials() {
        return numberOfTrials;
    }

    /**
     * Access the probability of success for this distribution.
     *
     * @return the probability of success.
     */
    public double getProbabilityOfSuccess() {
        return probabilityOfSuccess;
    }

    /**
     * {@inheritDoc}
     */
    public double probability(int x) {

        double ret;
        if (x < 0 || x > numberOfTrials) {
            ret = 0.0;
        } else {
            ret = FastMath.exp(SaddlePointExpansion.logBinomialProbability(x, numberOfTrials, probabilityOfSuccess,
                            1.0 - probabilityOfSuccess));
        }
        return ret;
    }

    /**
     * {@inheritDoc}
     */
    public double cumulativeProbability(int x) {

        double ret;
        if (x < 0) {
            ret = 0.0;
        } else if (x >= numberOfTrials) {
            ret = 1.0;
        } else {
            ret = 1.0 - Beta.regularizedBeta(probabilityOfSuccess, x + 1.0, numberOfTrials - x);
        }
        return ret;
    }

    @Override
    public double density(double x) {
        return 0;
    }

    @Override
    public double cumulativeProbability(double x) {

        double ret;
        if (x < 0) {
            ret = 0.0D;
        } else if (x >= this.numberOfTrials) {
            ret = 1.0D;
        } else {
            ret = 1.0D - Beta.regularizedBeta(this.probabilityOfSuccess, x + 1.0D, (this.numberOfTrials - x));
        }

        return ret;
    }

    @Override
    public double cumulativeProbability(double x0, double x1) throws NumberIsTooLargeException {
        return 0;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * For {@code n} trials and probability parameter {@code p}, the mean is
     * {@code n * p}.
     */
    public double getNumericalMean() {

        return numberOfTrials * probabilityOfSuccess;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * For {@code n} trials and probability parameter {@code p}, the variance is
     * {@code n * p * (1 - p)}.
     */
    public double getNumericalVariance() {

        final double p = probabilityOfSuccess;
        return numberOfTrials * p * (1 - p);
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The lower bound of the support is always 0 except for the probability
     * parameter {@code p = 1}.
     *
     * @return lower bound of the support (0 or the number of trials)
     */
    @Override
    public double getSupportLowerBound() {

        return probabilityOfSuccess < 1.0 ? 0 : numberOfTrials;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * The upper bound of the support is the number of trials except for the
     * probability parameter {@code p = 0}.
     *
     * @return upper bound of the support (number of trials or 0)
     */
    @Override
    public double getSupportUpperBound() {

        return probabilityOfSuccess > 0.0 ? numberOfTrials : 0;
    }

    @Override
    public boolean isSupportLowerBoundInclusive() {
        return false;
    }

    @Override
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


    private void ensureConsistent(int i) {
        probabilityOfSuccess = p.linearView().getDouble(i);
    }

    @Override
    public INDArray sample(int[] shape) {
        INDArray ret = Nd4j.createUninitialized(shape, Nd4j.order());
        return sample(ret);
    }

    @Override
    public INDArray sample(INDArray ret) {
        if (random.getStatePointer() != null) {
            if (p != null) {
                return Nd4j.getExecutioner()
                        .exec(new org.nd4j.linalg.api.ops.random.impl.BinomialDistributionEx(
                                        ret, numberOfTrials, p),
                                random);
            } else {
                return Nd4j.getExecutioner()
                        .exec(new org.nd4j.linalg.api.ops.random.impl.BinomialDistributionEx(
                                ret, numberOfTrials,
                                probabilityOfSuccess), random);
            }
        } else {
            Iterator<long[]> idxIter = new NdIndexIterator(ret.shape()); //For consistent values irrespective of c vs. fortran ordering
            long len = ret.length();
            if (p != null) {
                for (int i = 0; i < len; i++) {
                    long[] idx = idxIter.next();
                    org.apache.commons.math3.distribution.BinomialDistribution binomialDistribution =
                            new org.apache.commons.math3.distribution.BinomialDistribution(
                                    (RandomGenerator) Nd4j.getRandom(), numberOfTrials,
                                    p.getDouble(idx));
                    ret.putScalar(idx, binomialDistribution.sample());
                }
            } else {
                org.apache.commons.math3.distribution.BinomialDistribution binomialDistribution =
                        new org.apache.commons.math3.distribution.BinomialDistribution(
                                (RandomGenerator) Nd4j.getRandom(), numberOfTrials,
                                probabilityOfSuccess);
                for (int i = 0; i < len; i++) {
                    ret.putScalar(idxIter.next(), binomialDistribution.sample());
                }
            }
            return ret;
        }

    }
}
