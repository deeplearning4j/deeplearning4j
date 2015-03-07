/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.sampling;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.MathUtils;

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
    public static INDArray normal(RandomGenerator rng, INDArray mean, INDArray sigma) {
        INDArray iter = mean.linearView();
        INDArray sigmaLinear = sigma.linearView();
        for (int i = 0; i < iter.length(); i++) {
            RealDistribution reals = new NormalDistribution(rng, mean.getDouble(i), FastMath.sqrt(sigmaLinear.getDouble(i)), NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
            iter.putScalar(i, reals.sample());

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
    public static INDArray normal(RandomGenerator rng, INDArray mean, double sigma) {
        INDArray modify = Nd4j.create(mean.shape());
        INDArray iter = mean.linearView();
        INDArray linearModify = modify.linearView();

        double sqrt = FastMath.sqrt(sigma);
        for (int i = 0; i < iter.length(); i++) {
            double curr = iter.getDouble(i);
            RealDistribution reals = new NormalDistribution(rng, curr, sqrt, NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
            linearModify.putScalar(i, reals.sample());

        }
        return modify;
    }


    /**
     * Generate a binomial distribution based on the given rng,
     * a matrix of p values, and a max number.
     *
     * @param p   the p matrix to use
     * @param n   the n to use
     * @param rng the rng to use
     * @return a binomial distribution based on the one n, the passed in p values, and rng
     */
    public static INDArray binomial(INDArray p, int n, RandomGenerator rng) {
        INDArray p2 = p.dup();
        INDArray p2Linear = p2.linearView();
        for (int i = 0; i < p2.length(); i++) {
            p2Linear.putScalar(i, MathUtils.binomial(rng, n, p2Linear.getDouble(i)));
        }
        return p2;
    }


}
