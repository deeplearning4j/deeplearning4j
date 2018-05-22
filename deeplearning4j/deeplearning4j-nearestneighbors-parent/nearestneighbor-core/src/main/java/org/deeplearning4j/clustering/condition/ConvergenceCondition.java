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

package org.deeplearning4j.clustering.condition;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.deeplearning4j.clustering.iteration.IterationHistory;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.LessThan;

import java.io.Serializable;

@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor(access = AccessLevel.PROTECTED)
public class ConvergenceCondition implements ClusteringAlgorithmCondition, Serializable {

    private Condition convergenceCondition;
    private double pointsDistributionChangeRate;


    /**
     *
     * @param pointsDistributionChangeRate
     * @return
     */
    public static ConvergenceCondition distributionVariationRateLessThan(double pointsDistributionChangeRate) {
        Condition condition = new LessThan(pointsDistributionChangeRate);
        return new ConvergenceCondition(condition, pointsDistributionChangeRate);
    }


    /**
     *
     * @param iterationHistory
     * @return
     */
    public boolean isSatisfied(IterationHistory iterationHistory) {
        int iterationCount = iterationHistory.getIterationCount();
        if (iterationCount <= 1)
            return false;

        double variation = iterationHistory.getMostRecentClusterSetInfo().getPointLocationChange().get();
        variation /= iterationHistory.getMostRecentClusterSetInfo().getPointsCount();

        return convergenceCondition.apply(variation);
    }



}
