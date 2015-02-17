package org.deeplearning4j.clustering.algorithm.strategy;

import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public class FixedClusterCountStrategy extends BaseClusteringStrategy {
	public static int	defaultIterationCount	= 100;

	protected FixedClusterCountStrategy(Integer initialClusterCount, Class<? extends DistanceFunction> distanceFunction, boolean allowEmptyClusters) {
		super(ClusteringStrategyType.FIXED_CLUSTER_COUNT, initialClusterCount, distanceFunction, allowEmptyClusters);
	}

	public static FixedClusterCountStrategy setup(int clusterCount, Class<? extends DistanceFunction> distanceFunction) {
		return new FixedClusterCountStrategy(clusterCount, distanceFunction, false);
	}
	
	public boolean isOptimizationDefined() {
		return false;
	}
	public boolean isOptimizationApplicableNow(IterationHistory iterationHistory) {
		return false;
	}

}
