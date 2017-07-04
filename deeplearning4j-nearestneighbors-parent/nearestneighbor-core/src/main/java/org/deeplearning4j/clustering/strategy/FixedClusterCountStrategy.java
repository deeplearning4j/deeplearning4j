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

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.deeplearning4j.clustering.iteration.IterationHistory;

/**
 *
 */
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class FixedClusterCountStrategy extends BaseClusteringStrategy {


    protected FixedClusterCountStrategy(Integer initialClusterCount, String distanceFunction,
                    boolean allowEmptyClusters, boolean inverse) {
        super(ClusteringStrategyType.FIXED_CLUSTER_COUNT, initialClusterCount, distanceFunction, allowEmptyClusters,
                        inverse);
    }

    /**
     *
     * @param clusterCount
     * @param distanceFunction
     * @param inverse
     * @return
     */
    public static FixedClusterCountStrategy setup(int clusterCount, String distanceFunction, boolean inverse) {
        return new FixedClusterCountStrategy(clusterCount, distanceFunction, false, inverse);
    }

    /**
     * @return
     */
    @Override
    public boolean inverseDistanceCalculation() {
        return inverse;
    }

    public boolean isOptimizationDefined() {
        return false;
    }

    public boolean isOptimizationApplicableNow(IterationHistory iterationHistory) {
        return false;
    }

}
