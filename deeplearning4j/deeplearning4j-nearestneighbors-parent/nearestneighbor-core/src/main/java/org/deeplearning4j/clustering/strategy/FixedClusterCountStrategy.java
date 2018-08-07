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
