/*-
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

package org.deeplearning4j.clustering.strategy;

import org.deeplearning4j.clustering.condition.ClusteringAlgorithmCondition;
import org.deeplearning4j.clustering.iteration.IterationHistory;

/**
 *
 */
public interface ClusteringStrategy {

    /**
     *
     * @return
     */
    boolean inverseDistanceCalculation();

    /**
     *
     * @return
     */
    ClusteringStrategyType getType();

    /**
     *
     * @param type
     * @return
     */
    boolean isStrategyOfType(ClusteringStrategyType type);

    /**
     *
     * @return
     */
    Integer getInitialClusterCount();

    /**
     *
     * @return
     */
    String getDistanceFunction();

    /**
     *
     * @return
     */
    boolean isAllowEmptyClusters();

    /**
     *
     * @return
     */
    ClusteringAlgorithmCondition getTerminationCondition();

    /**
     *
     * @return
     */
    boolean isOptimizationDefined();

    /**
     *
     * @param iterationHistory
     * @return
     */
    boolean isOptimizationApplicableNow(IterationHistory iterationHistory);

    /**
     *
     * @param maxIterationCount
     * @return
     */
    BaseClusteringStrategy endWhenIterationCountEquals(int maxIterationCount);

    /**
     *
     * @param rate
     * @return
     */
    BaseClusteringStrategy endWhenDistributionVariationRateLessThan(double rate);

}
