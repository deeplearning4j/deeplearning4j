package org.nd4j.linalg.api.rng.distribution.factory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;

/**
 * Create a distribution
 *
 * @author Adam Gibson
 */
public interface DistributionFactory {

    /**
     * Create a distribution
     *
     * @param n the number of trials
     * @param p the probabilities
     * @return the biniomial distribution with the given parameters
     */
    Distribution createBinomial(int n, INDArray p);

    /**
     * Create a distribution
     *
     * @param n the number of trials
     * @param p the probabilities
     * @return the biniomial distribution with the given parameters
     */
    Distribution createBinomial(int n, double p);

    /**
     * Create  a normal distribution
     * with the given mean and std
     *
     * @param mean the mean
     * @param std  the standard deviation
     * @return the distribution with the given
     * mean and standard deviation
     */
    Distribution createNormal(INDArray mean, double std);

    /**
     * Create  a normal distribution
     * with the given mean and std
     *
     * @param mean the mean
     * @param std  the stnadard deviation
     * @return the distribution with the given
     * mean and standard deviation
     */
    Distribution createNormal(double mean, double std);

    /**
     * Create a uniform distribution with the
     * given min and max
     *
     * @param min the min
     * @param max the max
     * @return the uniform distribution
     */
    Distribution createUniform(double min, double max);

}
