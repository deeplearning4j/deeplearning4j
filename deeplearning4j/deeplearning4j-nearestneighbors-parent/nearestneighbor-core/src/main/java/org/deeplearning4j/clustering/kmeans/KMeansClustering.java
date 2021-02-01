/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.clustering.kmeans;

import org.deeplearning4j.clustering.algorithm.BaseClusteringAlgorithm;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.strategy.ClusteringStrategy;
import org.deeplearning4j.clustering.strategy.FixedClusterCountStrategy;


public class KMeansClustering extends BaseClusteringAlgorithm {

    private static final long serialVersionUID = 8476951388145944776L;
    private static final double VARIATION_TOLERANCE= 1e-4;


    /**
     *
     * @param clusteringStrategy
     */
    protected KMeansClustering(ClusteringStrategy clusteringStrategy, boolean useKMeansPlusPlus) {
        super(clusteringStrategy, useKMeansPlusPlus);
    }

    /**
     * Setup a kmeans instance
     * @param clusterCount the number of clusters
     * @param maxIterationCount the max number of iterations
     *                          to run kmeans
     * @param distanceFunction the distance function to use for grouping
     * @return
     */
    public static KMeansClustering setup(int clusterCount, int maxIterationCount, Distance distanceFunction,
                    boolean inverse, boolean useKMeansPlusPlus) {
        ClusteringStrategy clusteringStrategy =
                        FixedClusterCountStrategy.setup(clusterCount, distanceFunction, inverse);
        clusteringStrategy.endWhenIterationCountEquals(maxIterationCount);
        return new KMeansClustering(clusteringStrategy, useKMeansPlusPlus);
    }

    /**
     *
     * @param clusterCount
     * @param minDistributionVariationRate
     * @param distanceFunction
     * @param allowEmptyClusters
     * @return
     */
    public static KMeansClustering setup(int clusterCount, double minDistributionVariationRate, Distance distanceFunction,
                    boolean inverse, boolean allowEmptyClusters, boolean useKMeansPlusPlus) {
        ClusteringStrategy clusteringStrategy = FixedClusterCountStrategy.setup(clusterCount, distanceFunction, inverse)
                        .endWhenDistributionVariationRateLessThan(minDistributionVariationRate);
        return new KMeansClustering(clusteringStrategy, useKMeansPlusPlus);
    }


    /**
     * Setup a kmeans instance
     * @param clusterCount the number of clusters
     * @param maxIterationCount the max number of iterations
     *                          to run kmeans
     * @param distanceFunction the distance function to use for grouping
     * @return
     */
    public static KMeansClustering setup(int clusterCount, int maxIterationCount, Distance distanceFunction, boolean useKMeansPlusPlus) {
        return setup(clusterCount, maxIterationCount, distanceFunction, false, useKMeansPlusPlus);
    }

    /**
     *
     * @param clusterCount
     * @param minDistributionVariationRate
     * @param distanceFunction
     * @param allowEmptyClusters
     * @return
     */
    public static KMeansClustering setup(int clusterCount, double minDistributionVariationRate, Distance distanceFunction,
                    boolean allowEmptyClusters, boolean useKMeansPlusPlus) {
        ClusteringStrategy clusteringStrategy = FixedClusterCountStrategy.setup(clusterCount, distanceFunction, false);
        clusteringStrategy.endWhenDistributionVariationRateLessThan(minDistributionVariationRate);
        return new KMeansClustering(clusteringStrategy, useKMeansPlusPlus);
    }

    public static KMeansClustering setup(int clusterCount, Distance distanceFunction,
                                         boolean allowEmptyClusters, boolean useKMeansPlusPlus) {
        ClusteringStrategy clusteringStrategy = FixedClusterCountStrategy.setup(clusterCount, distanceFunction, false);
        clusteringStrategy.endWhenDistributionVariationRateLessThan(VARIATION_TOLERANCE);
        return new KMeansClustering(clusteringStrategy, useKMeansPlusPlus);
    }

}
