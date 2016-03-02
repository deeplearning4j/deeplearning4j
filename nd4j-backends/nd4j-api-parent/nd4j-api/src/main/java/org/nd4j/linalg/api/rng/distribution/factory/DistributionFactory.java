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
