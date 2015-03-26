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

package org.arbiter.clustering.algorithm.condition;

import org.arbiter.clustering.algorithm.iteration.IterationHistory;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.LessThan;

public class ConvergenceCondition implements ClusteringAlgorithmCondition {

	private Condition	convergenceCondition;
	private double pointsDistributionChangeRate;
	
	
	protected ConvergenceCondition(Condition varianceVariationCondition, double pointsDistributionChangeRate) {
		super();
		this.convergenceCondition = varianceVariationCondition;
		this.pointsDistributionChangeRate = pointsDistributionChangeRate;
	}

	public static ConvergenceCondition distributionVariationRateLessThan(double pointsDistributionChangeRate) {
		Condition condition = new LessThan(pointsDistributionChangeRate);
		return new ConvergenceCondition(condition, pointsDistributionChangeRate);
	}

	
	public boolean isSatisfied(IterationHistory iterationHistory) {
		int iterationCount = iterationHistory.getIterationCount();
		if( iterationCount<=1 )
			return false;
		
		double variation = iterationHistory.getMostRecentClusterSetInfo().getPointLocationChange().get();
		variation /= iterationHistory.getMostRecentClusterSetInfo().getPointsCount();
		
		return convergenceCondition.apply(variation);
	}

	

}
