package org.deeplearning4j.clustering.algorithm.strategy;

import org.deeplearning4j.clustering.algorithm.condition.ClusteringEndCondition;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public class MinimizeInClusterVarianceStrategy extends BaseClusteringStrategy {

	protected MinimizeInClusterVarianceStrategy(Integer initialClusterCount, Class<? extends DistanceFunction> distanceFunction,
			ClusteringEndCondition clusteringEndCondition, boolean allowEmptyClusters) {
		super(StrategyType.MINIMIZE_INTRA_CLUSTER_VARIANCE, initialClusterCount, distanceFunction, clusteringEndCondition, allowEmptyClusters);
	}

	public static MinimizeInClusterVarianceStrategy setup(int initialClusterCount, Class<? extends DistanceFunction> distanceFunction,
			double targetAveragePointDistanceToClusterCenter) {
		return setup(initialClusterCount, distanceFunction, targetAveragePointDistanceToClusterCenter, false);
	}

	public static MinimizeInClusterVarianceStrategy setup(int initialClusterCount, Class<? extends DistanceFunction> distanceFunction,
			double targetAveragePointDistanceToClusterCenter, boolean allowEmptyClusters) {
		ClusteringEndCondition endCondition = new ClusteringEndCondition()
				.averagePointDistanceToClusterCenterLessThan(targetAveragePointDistanceToClusterCenter);

		return new MinimizeInClusterVarianceStrategy(initialClusterCount, distanceFunction, endCondition, allowEmptyClusters);
	}

}
