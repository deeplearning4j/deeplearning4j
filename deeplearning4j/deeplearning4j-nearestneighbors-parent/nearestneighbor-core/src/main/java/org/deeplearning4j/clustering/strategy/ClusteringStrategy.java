/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
