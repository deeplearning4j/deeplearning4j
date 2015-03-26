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

package org.arbiter.clustering.kmeans;

import org.arbiter.clustering.algorithm.BaseClusteringAlgorithm;
import org.arbiter.clustering.algorithm.strategy.ClusteringStrategy;
import org.arbiter.clustering.algorithm.strategy.FixedClusterCountStrategy;


/**
 * 
 * @author Julien Roch
 *
 */
public class KMeansClustering extends BaseClusteringAlgorithm {

	private static final long	serialVersionUID	= 8476951388145944776L;
	
	

	protected KMeansClustering(ClusteringStrategy clusteringStrategy) {
		super(clusteringStrategy);
	}



	public static KMeansClustering setup(int clusterCount, int maxIterationCount,String distanceFunction) {
		ClusteringStrategy clusteringStrategy = FixedClusterCountStrategy.setup(clusterCount, distanceFunction);
		clusteringStrategy.endWhenIterationCountEquals(maxIterationCount);
		return new KMeansClustering(clusteringStrategy);
	}

	public static KMeansClustering setup(int clusterCount, double minDistributionVariationRate, String distanceFunction, boolean allowEmptyClusters) {
		ClusteringStrategy clusteringStrategy = FixedClusterCountStrategy.setup(clusterCount, distanceFunction);
		clusteringStrategy.endWhenDistributionVariationRateLessThan(minDistributionVariationRate);
		return new KMeansClustering(clusteringStrategy);
	}
	

}
