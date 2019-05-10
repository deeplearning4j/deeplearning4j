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

package org.nd4j.linalg.api.rng.distribution.factory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.LogNormalDistribution;
import org.nd4j.linalg.api.rng.distribution.impl.OrthogonalDistribution;
import org.nd4j.linalg.api.rng.distribution.impl.TruncatedNormalDistribution;

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

    /**
     * Creates a log-normal distribution
     *
     * @param mean
     * @param std
     * @return
     */
    Distribution createLogNormal(double mean, double std);

    /**
     * Creates truncated normal distribution
     *
     * @param mean
     * @param std
     * @return
     */
    Distribution createTruncatedNormal(double mean, double std);

    /**
     * Creates orthogonal distribution
     *
     * @param gain
     * @return
     */
    Distribution createOrthogonal(double gain);

    /**
     * Creates constant distribution
     *
     * @param value
     * @return
     */
    Distribution createConstant(double value);
}
