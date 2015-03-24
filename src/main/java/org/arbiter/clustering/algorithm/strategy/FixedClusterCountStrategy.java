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

package org.arbiter.clustering.algorithm.strategy;

import org.arbiter.clustering.algorithm.iteration.IterationHistory;

public class FixedClusterCountStrategy extends BaseClusteringStrategy {
	public static int	defaultIterationCount	= 100;

	protected FixedClusterCountStrategy(Integer initialClusterCount, String distanceFunction, boolean allowEmptyClusters) {
		super(ClusteringStrategyType.FIXED_CLUSTER_COUNT, initialClusterCount, distanceFunction, allowEmptyClusters);
	}

	public static FixedClusterCountStrategy setup(int clusterCount,String distanceFunction) {
		return new FixedClusterCountStrategy(clusterCount, distanceFunction, false);
	}
	
	public boolean isOptimizationDefined() {
		return false;
	}
	public boolean isOptimizationApplicableNow(IterationHistory iterationHistory) {
		return false;
	}

}
