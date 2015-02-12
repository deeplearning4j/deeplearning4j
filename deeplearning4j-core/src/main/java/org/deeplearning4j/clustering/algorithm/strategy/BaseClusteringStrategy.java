package org.deeplearning4j.clustering.algorithm.strategy;

import org.deeplearning4j.clustering.algorithm.condition.ClusteringEndCondition;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public abstract class BaseClusteringStrategy implements ClusteringStrategy {

	protected StrategyType						type;
	protected Integer							initialClusterCount;

	protected Class<? extends DistanceFunction>	distanceFunction;
	protected ClusteringEndCondition			clusteringEndCondition;

	protected boolean							allowEmptyClusters;

	
	protected BaseClusteringStrategy(StrategyType type, Integer initialClusterCount, Class<? extends DistanceFunction> distanceFunction,
			ClusteringEndCondition clusteringEndCondition, boolean allowEmptyClusters) {
		super();
		this.type = type;
		this.initialClusterCount = initialClusterCount;
		this.distanceFunction = distanceFunction;
		this.clusteringEndCondition = clusteringEndCondition;
		this.allowEmptyClusters = allowEmptyClusters;
	}

	
	public boolean isStrategyOfType(StrategyType type) {
		return type.equals(this.type);
	}

	public Integer getInitialClusterCount() {
		return initialClusterCount;
	}

	public void setInitialClusterCount(Integer clusterCount) {
		this.initialClusterCount = clusterCount;
	}

	public ClusteringEndCondition getClusteringEndCondition() {
		return clusteringEndCondition;
	}

	public void setClusteringEndCondition(ClusteringEndCondition endCondition) {
		this.clusteringEndCondition = endCondition;
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



	public StrategyType getType() {
		return type;
	}



	public void setType(StrategyType type) {
		this.type = type;
	}

}
