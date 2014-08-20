package org.deeplearning4j.linalg.sampling;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.MathUtils;

/**
 * Static methods for sampling from an ndarray
 *
 * @author Adam Gibson
 */
public class Sampling {




    /**
     * A uniform sample ranging from 0 to sigma.
     *
     * @param rng   the rng to use
     * @param mean, the matrix mean from which to generate values from
     * @param sigma the standard deviation to use to generate the gaussian noise
     * @return a uniform sample of the given shape and size
     * <p/>
     * with numbers between 0 and 1
     */
    public static INDArray  normal(RandomGenerator rng, INDArray mean, INDArray sigma) {
        INDArray iter = mean.reshape(new int[]{1,mean.length()}).dup();
        INDArray sigmaLinear = sigma.ravel();
        for(int i = 0; i < iter.length(); i++) {
            RealDistribution reals = new NormalDistribution(rng,(double) mean.getScalar(i).element(), FastMath.sqrt((double) sigmaLinear.getScalar(i).element()),NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
            iter.putScalar(i,reals.sample());

        }
        return iter.reshape(mean.shape());
    }

    /**
     * A uniform sample ranging from 0 to sigma.
     *
     * @param rng   the rng to use
     * @param mean, the matrix mean from which to generate values from
     * @param sigma the standard deviation to use to generate the gaussian noise
     * @return a uniform sample of the given shape and size
     * <p/>
     * with numbers between 0 and 1
     */
    public static INDArray  normal(RandomGenerator rng, INDArray mean, double sigma) {
        INDArray iter = mean.ravel().dup();
        for(int i = 0; i < iter.length(); i++) {
            RealDistribution reals = new NormalDistribution(rng,(double) mean.getScalar(i).element(), FastMath.sqrt(sigma),NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
            iter.putScalar(i,reals.sample());

        }
        return iter.reshape(mean.shape());
    }


    /**
     * Generate a binomial distribution based on the given rng,
     * a matrix of p values, and a max number.
     * @param p the p matrix to use
     * @param n the n to use
     * @param rng the rng to use
     * @return a binomial distribution based on the one n, the passed in p values, and rng
     */
    public static INDArray binomial(INDArray p,int n,RandomGenerator rng) {
        INDArray ret = p.ravel().dup();
        INDArray p2 = p.ravel();
        for(int i = 0; i < ret.length(); i++) {
            ret.putScalar(i, MathUtils.binomial(rng, n, (Double) p2.getScalar(i).element()));
        }
        return ret.reshape(p.shape());
    }


}
