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

import org.arbiter.clustering.algorithm.condition.ClusteringAlgorithmCondition;
import org.arbiter.clustering.algorithm.condition.FixedIterationCountCondition;
import org.arbiter.clustering.algorithm.condition.ConvergenceCondition;

public abstract class BaseClusteringStrategy implements ClusteringStrategy {

	protected ClusteringStrategyType type;
	protected Integer initialClusterCount;
	protected ClusteringAlgorithmCondition optimizationPhaseCondition;
	protected ClusteringAlgorithmCondition terminationCondition;

	protected String	distanceFunction;

	protected boolean allowEmptyClusters;

	protected BaseClusteringStrategy(ClusteringStrategyType type, Integer initialClusterCount, String distanceFunction,
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

	public String getDistanceFunction() {
		return distanceFunction;
	}

	public void setDistanceFunction(String distanceFunction) {
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
