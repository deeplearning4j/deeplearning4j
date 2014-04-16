package org.deeplearning4j.distributions;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;

public class Distributions {





    /**
     * Returns a exponential distribution
     * with a mean of 0 and a standard deviation of std
     * @param rng the rng to use
     * @param mean the standard mean to use
     * @return a normal distribution with a mean of 0 and a standard deviation of 1
     */
    public static RealDistribution exponential(RandomGenerator rng,double mean) {
        return  new ExponentialDistribution(rng,1.0,ExponentialDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    /**
	 * Returns a normal distribution 
	 * with a mean of 0 and a standard deviation of std
	 * @param rng the rng to use
	 * @param std the standard deviation to use
	 * @return a normal distribution with a mean of 0 and a standard deviation of 1
	 */
	public static RealDistribution normal(RandomGenerator rng,double std) {
		return  new NormalDistribution(rng,0,std,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
	}
	
	/**
	 * Returns a uniform distribution with a
	 * min of -fanIn and a max of positivefanIn
	 * @param rng the rng to use
	 * @param fanIn the fanin to use
	 * @return a uniform distribution with a min of -fanIn
	 * and a max of fanIn
	 */
	public static RealDistribution uniform(RandomGenerator rng,double fanIn) {
		fanIn = Math.abs(fanIn);
		return new UniformRealDistribution(rng,-fanIn,fanIn);
	}
	
	/**
	 * Returns a uniform distribution 
	 * based on the number of ins and outs
	 * @param rng the rng to use
	 * @param nIn the number of inputs
	 * @param nOut the number of outputs
	 * @return
	 */
	public static RealDistribution uniform(RandomGenerator rng,int nIn,int nOut) {
		double fanIn = -4 * Math.sqrt(6. / (nOut + nIn));
		return uniform(rng,fanIn);
	}
	
	/**
	 * Returns a uniform distribution with a
	 * min of -fanIn and a max of positivefanIn
	 * @param rng the rng to use
	 * @return a uniform distribution with a min of -fanIn
	 * and a max of fanIn
	 */
	public static RealDistribution uniform(RandomGenerator rng) {
		double fanIn = 0.1;
		return new UniformRealDistribution(rng,-fanIn,fanIn);
	}

}
