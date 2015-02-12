package org.deeplearning4j.clustering.algorithm.strategy;

import org.deeplearning4j.clustering.algorithm.condition.ClusteringEndCondition;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public class FixedClusterCountStrategy extends BaseClusteringStrategy {
	public static int	defaultIterationCount	= 100;

	protected FixedClusterCountStrategy(Integer initialClusterCount, Class<? extends DistanceFunction> distanceFunction,
			ClusteringEndCondition clusteringEndCondition, boolean allowEmptyClusters) {
		super(StrategyType.FIXED_CLUSTER_COUNT, initialClusterCount, distanceFunction, clusteringEndCondition, allowEmptyClusters);
	}

	public static FixedClusterCountStrategy setup(int clusterCount, Class<? extends DistanceFunction> distanceFunction, int maxIterationCount) {
		return setup(clusterCount, distanceFunction, maxIterationCount, false);
	}
	
	public static FixedClusterCountStrategy setup(int clusterCount, Class<? extends DistanceFunction> distanceFunction, int maxIterationCount, boolean allowEmptyClusters) {
		ClusteringEndCondition endCondition = new ClusteringEndCondition().iterationCountGreaterThan(maxIterationCount);
		FixedClusterCountStrategy strategy = new FixedClusterCountStrategy(clusterCount, distanceFunction, endCondition, allowEmptyClusters);
		return strategy;
	}
}
