/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.clustering.algorithm.condition;

import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;
import org.deeplearning4j.clustering.algorithm.strategy.FixedClusterCountStrategy;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;

import java.io.Serializable;

public class FixedIterationCountCondition implements ClusteringAlgorithmCondition, Serializable {

	private Condition	iterationCountCondition;

	protected FixedIterationCountCondition() {
		// no-op for serialization only
	}

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
