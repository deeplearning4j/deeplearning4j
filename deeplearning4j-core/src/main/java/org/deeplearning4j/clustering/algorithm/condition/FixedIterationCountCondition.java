package org.deeplearning4j.clustering.algorithm.condition;

import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;

public class FixedIterationCountCondition implements ClusteringAlgorithmCondition {

	private Condition	iterationCountCondition;

	protected FixedIterationCountCondition(int initialClusterCount) {
		iterationCountCondition = new GreaterThanOrEqual(initialClusterCount);
	}

	public static FixedIterationCountCondition iterationCountGreaterThan(int iterationCount) {
		return new FixedIterationCountCondition(iterationCount);
	}

	public boolean isSatisfied(IterationHistory iterationHistory) {
		return iterationCountCondition.apply(iterationHistory == null ? 0 : iterationHistory.getIterationCount());
	}
	
}
