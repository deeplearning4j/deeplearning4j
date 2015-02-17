package org.deeplearning4j.clustering.algorithm.strategy;

import org.deeplearning4j.clustering.algorithm.condition.ClusteringAlgorithmCondition;
import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public interface ClusteringStrategy {

	ClusteringStrategyType getType();
	boolean isStrategyOfType(ClusteringStrategyType type);
	
	Integer getInitialClusterCount();
	
	Class<? extends DistanceFunction> getDistanceFunction();

	boolean isAllowEmptyClusters();

	ClusteringAlgorithmCondition getTerminationCondition();
	
	boolean isOptimizationDefined();
	boolean isOptimizationApplicableNow(IterationHistory iterationHistory);
	
	BaseClusteringStrategy endWhenIterationCountEquals(int maxIterationCount);
	BaseClusteringStrategy endWhenDistributionVariationRateLessThan(double rate);
	
}
