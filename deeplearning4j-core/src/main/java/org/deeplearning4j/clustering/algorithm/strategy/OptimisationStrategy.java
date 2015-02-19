package org.deeplearning4j.clustering.algorithm.strategy;

import org.deeplearning4j.clustering.algorithm.condition.ClusteringAlgorithmCondition;
import org.deeplearning4j.clustering.algorithm.condition.ConvergenceCondition;
import org.deeplearning4j.clustering.algorithm.condition.FixedIterationCountCondition;
import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;
import org.deeplearning4j.clustering.algorithm.optimisation.ClusteringOptimization;
import org.deeplearning4j.clustering.algorithm.optimisation.ClusteringOptimizationType;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public class OptimisationStrategy extends BaseClusteringStrategy {
	public static int									defaultIterationCount	= 100;

	private ClusteringOptimization						clusteringOptimisation;
	private ClusteringAlgorithmCondition				clusteringOptimisationApplicationCondition;

	protected OptimisationStrategy(int initialClusterCount, Class<? extends DistanceFunction> distanceFunction) {
		super(ClusteringStrategyType.OPTIMIZATION, initialClusterCount, distanceFunction, false);
	}

	public static OptimisationStrategy setup(int initialClusterCount, Class<? extends DistanceFunction> distanceFunction) {
		return new OptimisationStrategy(initialClusterCount, distanceFunction);
	}
	
	public OptimisationStrategy optimize(ClusteringOptimizationType type, double value) {
		clusteringOptimisation = new ClusteringOptimization(type, value);
		return this;
	}
	
	public OptimisationStrategy optimizeWhenIterationCountMultipleOf(int value) {
		clusteringOptimisationApplicationCondition = FixedIterationCountCondition.iterationCountGreaterThan(value);
		return this;
	}
	
	public OptimisationStrategy optimizeWhenPointDistributionVariationRateLessThan(double rate) {
		clusteringOptimisationApplicationCondition = ConvergenceCondition.distributionVariationRateLessThan(rate);
		return this;
	}
	

	public double getClusteringOptimizationValue() {
		return clusteringOptimisation.getValue();
	}
	
	public boolean isClusteringOptimizationType(ClusteringOptimizationType type) {
		return clusteringOptimisation!=null && clusteringOptimisation.getType().equals(type);
	}

	public boolean isOptimizationDefined() {
		return clusteringOptimisation != null;
	}

	public boolean isOptimizationApplicableNow(IterationHistory iterationHistory) {
		return clusteringOptimisationApplicationCondition!=null && clusteringOptimisationApplicationCondition.isSatisfied(iterationHistory);
	}

}
