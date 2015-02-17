package org.deeplearning4j.clustering.algorithm.condition;

import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;

public interface ClusteringAlgorithmCondition {

	boolean isSatisfied(IterationHistory iterationHistory);
	
}
