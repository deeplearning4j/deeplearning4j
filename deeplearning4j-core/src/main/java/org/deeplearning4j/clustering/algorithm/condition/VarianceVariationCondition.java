package org.deeplearning4j.clustering.algorithm.condition;

import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.LessThan;

public class VarianceVariationCondition implements ClusteringAlgorithmCondition {

	private Condition	varianceVariationCondition;
	private int			period;
	
	
	protected VarianceVariationCondition(Condition varianceVariationCondition, int period) {
		super();
		this.varianceVariationCondition = varianceVariationCondition;
		this.period = period;
	}

	public static VarianceVariationCondition varianceVariationLessThan(double varianceVariation, int period) {
		Condition condition = new LessThan(varianceVariation);
		return new VarianceVariationCondition(condition, period);
	}

	
	public boolean isSatisfied(IterationHistory iterationHistory) {
		if( iterationHistory.getIterationCount()<=period )
			return false;
		
		for( int i=0, j=iterationHistory.getIterationCount();i<period;i++) {
			double variation = iterationHistory.getIterationInfo(j-i).getClusterSetInfo().getPointDistanceFromClusterVariance();
			variation -= iterationHistory.getIterationInfo(j-i-1).getClusterSetInfo().getPointDistanceFromClusterVariance();
			variation /= iterationHistory.getIterationInfo(j-i-1).getClusterSetInfo().getPointDistanceFromClusterVariance();
			if( !varianceVariationCondition.apply(variation) )
				return false;
		}
		
		return true;
	}

	

}
