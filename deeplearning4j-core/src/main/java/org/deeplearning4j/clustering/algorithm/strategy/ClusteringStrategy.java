package org.deeplearning4j.clustering.algorithm.strategy;

import org.deeplearning4j.clustering.algorithm.condition.ClusteringEndCondition;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public interface ClusteringStrategy {

	Integer getInitialClusterCount();

	Class<? extends DistanceFunction> getDistanceFunction();

	boolean isAllowEmptyClusters();

	ClusteringEndCondition getClusteringEndCondition();
	
	StrategyType getType();
	boolean isStrategyOfType(StrategyType type);
}
