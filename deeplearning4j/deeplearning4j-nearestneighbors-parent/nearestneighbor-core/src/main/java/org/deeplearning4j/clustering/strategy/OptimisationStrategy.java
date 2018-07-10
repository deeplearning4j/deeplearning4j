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
import org.deeplearning4j.clustering.condition.ConvergenceCondition;
import org.deeplearning4j.clustering.condition.FixedIterationCountCondition;
import org.deeplearning4j.clustering.iteration.IterationHistory;
import org.deeplearning4j.clustering.optimisation.ClusteringOptimization;
import org.deeplearning4j.clustering.optimisation.ClusteringOptimizationType;

public class OptimisationStrategy extends BaseClusteringStrategy {
    public static int defaultIterationCount = 100;

    private ClusteringOptimization clusteringOptimisation;
    private ClusteringAlgorithmCondition clusteringOptimisationApplicationCondition;

    protected OptimisationStrategy() {
        super();
    }

    protected OptimisationStrategy(int initialClusterCount, String distanceFunction) {
        super(ClusteringStrategyType.OPTIMIZATION, initialClusterCount, distanceFunction, false);
    }

    public static OptimisationStrategy setup(int initialClusterCount, String distanceFunction) {
        return new OptimisationStrategy(initialClusterCount, distanceFunction);
    }

    public OptimisationStrategy optimize(ClusteringOptimizationType type, double value) {
        clusteringOptimisation = new ClusteringOptimization(type, value);
        return this;
    }

    public OptimisationStrategy optimizeWhenIterationCountMultipleOf(int value) {
        clusteringOptimisationApplicationCondition = FixedIterationCountCondition.iterationCountGreaterThan(value);
        return this;
    }

    public OptimisationStrategy optimizeWhenPointDistributionVariationRateLessThan(double rate) {
        clusteringOptimisationApplicationCondition = ConvergenceCondition.distributionVariationRateLessThan(rate);
        return this;
    }


    public double getClusteringOptimizationValue() {
        return clusteringOptimisation.getValue();
    }

    public boolean isClusteringOptimizationType(ClusteringOptimizationType type) {
        return clusteringOptimisation != null && clusteringOptimisation.getType().equals(type);
    }

    public boolean isOptimizationDefined() {
        return clusteringOptimisation != null;
    }

    public boolean isOptimizationApplicableNow(IterationHistory iterationHistory) {
        return clusteringOptimisationApplicationCondition != null
                        && clusteringOptimisationApplicationCondition.isSatisfied(iterationHistory);
    }

}
