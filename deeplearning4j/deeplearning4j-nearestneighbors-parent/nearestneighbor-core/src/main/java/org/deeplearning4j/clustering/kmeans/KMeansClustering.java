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

package org.deeplearning4j.clustering.kmeans;

import org.deeplearning4j.clustering.algorithm.BaseClusteringAlgorithm;
import org.deeplearning4j.clustering.strategy.ClusteringStrategy;
import org.deeplearning4j.clustering.strategy.FixedClusterCountStrategy;


/**
 *
 * @author Julien Roch
 *
 */
public class KMeansClustering extends BaseClusteringAlgorithm {

    private static final long serialVersionUID = 8476951388145944776L;


    /**
     *
     * @param clusteringStrategy
     */
    protected KMeansClustering(ClusteringStrategy clusteringStrategy) {
        super(clusteringStrategy);
    }

    /**
     * Setup a kmeans instance
     * @param clusterCount the number of clusters
     * @param maxIterationCount the max number of iterations
     *                          to run kmeans
     * @param distanceFunction the distance function to use for grouping
     * @return
     */
    public static KMeansClustering setup(int clusterCount, int maxIterationCount, String distanceFunction,
                    boolean inverse) {
        ClusteringStrategy clusteringStrategy =
                        FixedClusterCountStrategy.setup(clusterCount, distanceFunction, inverse);
        clusteringStrategy.endWhenIterationCountEquals(maxIterationCount);
        return new KMeansClustering(clusteringStrategy);
    }

    /**
     *
     * @param clusterCount
     * @param minDistributionVariationRate
     * @param distanceFunction
     * @param allowEmptyClusters
     * @return
     */
    public static KMeansClustering setup(int clusterCount, double minDistributionVariationRate, String distanceFunction,
                    boolean inverse, boolean allowEmptyClusters) {
        ClusteringStrategy clusteringStrategy = FixedClusterCountStrategy.setup(clusterCount, distanceFunction, inverse)
                        .endWhenDistributionVariationRateLessThan(minDistributionVariationRate);
        return new KMeansClustering(clusteringStrategy);
    }


    /**
     * Setup a kmeans instance
     * @param clusterCount the number of clusters
     * @param maxIterationCount the max number of iterations
     *                          to run kmeans
     * @param distanceFunction the distance function to use for grouping
     * @return
     */
    public static KMeansClustering setup(int clusterCount, int maxIterationCount, String distanceFunction) {
        return setup(clusterCount, maxIterationCount, distanceFunction, false);
    }

    /**
     *
     * @param clusterCount
     * @param minDistributionVariationRate
     * @param distanceFunction
     * @param allowEmptyClusters
     * @return
     */
    public static KMeansClustering setup(int clusterCount, double minDistributionVariationRate, String distanceFunction,
                    boolean allowEmptyClusters) {
        ClusteringStrategy clusteringStrategy = FixedClusterCountStrategy.setup(clusterCount, distanceFunction, false);
        clusteringStrategy.endWhenDistributionVariationRateLessThan(minDistributionVariationRate);
        return new KMeansClustering(clusteringStrategy);
    }


}
