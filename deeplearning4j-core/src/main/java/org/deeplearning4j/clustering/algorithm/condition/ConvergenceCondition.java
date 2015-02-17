package org.deeplearning4j.clustering.algorithm.condition;

import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.LessThan;

public class ConvergenceCondition implements ClusteringAlgorithmCondition {

	private Condition	convergenceCondition;
	private double		pointsDistributionChangeRate;
	
	
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
