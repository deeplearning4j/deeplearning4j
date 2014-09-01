package org.deeplearning4j.linalg.sampling;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
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
            RealDistribution reals = new NormalDistribution(rng, mean.get(i), FastMath.sqrt((double) sigmaLinear.get(i)),NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
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
        INDArray modify = NDArrays.create(mean.shape());
        INDArray iter = mean.linearView();
        INDArray linearModify = modify.linearView();

        double sqrt = FastMath.sqrt(sigma);
        for(int i = 0; i < iter.length(); i++) {
            double curr = iter.get(i);
            RealDistribution reals = new NormalDistribution(rng,curr, sqrt,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
            linearModify.putScalar(i,reals.sample());

        }
        return modify;
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
        INDArray p2 = p.dup();
        INDArray p2Linear = p2.linearView();
        for(int i = 0; i < p2.length(); i++) {
            p2Linear.putScalar(i, MathUtils.binomial(rng, n,p2Linear.get(i)));
        }
        return p2;
    }


}
