package org.deeplearning4j.clustering.algorithm.strategy;

import org.deeplearning4j.clustering.algorithm.condition.ClusteringAlgorithmCondition;
import org.deeplearning4j.clustering.algorithm.condition.ConvergenceCondition;
import org.deeplearning4j.clustering.algorithm.condition.FixedIterationCountCondition;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public abstract class BaseClusteringStrategy implements ClusteringStrategy {

	protected ClusteringStrategyType			type;
	protected Integer							initialClusterCount;
	protected ClusteringAlgorithmCondition		optimizationPhaseCondition;
	protected ClusteringAlgorithmCondition		terminationCondition;

	protected Class<? extends DistanceFunction>	distanceFunction;

	protected boolean							allowEmptyClusters;

	protected BaseClusteringStrategy(ClusteringStrategyType type, Integer initialClusterCount, Class<? extends DistanceFunction> distanceFunction,
			boolean allowEmptyClusters) {
		super();
		this.type = type;
		this.initialClusterCount = initialClusterCount;
		this.distanceFunction = distanceFunction;
		this.allowEmptyClusters = allowEmptyClusters;
	}

	public BaseClusteringStrategy endWhenIterationCountEquals(int maxIterationCount) {
		setTerminationCondition(FixedIterationCountCondition.iterationCountGreaterThan(maxIterationCount));
		return this;
	}

	public BaseClusteringStrategy endWhenDistributionVariationRateLessThan(double rate) {
		setTerminationCondition(ConvergenceCondition.distributionVariationRateLessThan(rate));
		return this;
	}

	public boolean isStrategyOfType(ClusteringStrategyType type) {
		return type.equals(this.type);
	}

	public Integer getInitialClusterCount() {
		return initialClusterCount;
	}

	public void setInitialClusterCount(Integer clusterCount) {
		this.initialClusterCount = clusterCount;
	}

	public Class<? extends DistanceFunction> getDistanceFunction() {
		return distanceFunction;
	}

	public void setDistanceFunction(Class<? extends DistanceFunction> distanceFunction) {
		this.distanceFunction = distanceFunction;
	}

	public boolean isAllowEmptyClusters() {
		return allowEmptyClusters;
	}

	public void setAllowEmptyClusters(boolean allowEmptyClusters) {
		this.allowEmptyClusters = allowEmptyClusters;
	}

	public ClusteringStrategyType getType() {
		return type;
	}

	protected void setType(ClusteringStrategyType type) {
		this.type = type;
	}

	public ClusteringAlgorithmCondition getOptimizationPhaseCondition() {
		return optimizationPhaseCondition;
	}

	protected void setOptimizationPhaseCondition(ClusteringAlgorithmCondition optimizationPhaseCondition) {
		this.optimizationPhaseCondition = optimizationPhaseCondition;
	}

	public ClusteringAlgorithmCondition getTerminationCondition() {
		return terminationCondition;
	}

	protected void setTerminationCondition(ClusteringAlgorithmCondition terminationCondition) {
		this.terminationCondition = terminationCondition;
	}

}
