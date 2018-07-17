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

package org.deeplearning4j.clustering.condition;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.deeplearning4j.clustering.iteration.IterationHistory;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;

import java.io.Serializable;

/**
 *
 */
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class FixedIterationCountCondition implements ClusteringAlgorithmCondition, Serializable {

    private Condition iterationCountCondition;

    protected FixedIterationCountCondition(int initialClusterCount) {
        iterationCountCondition = new GreaterThanOrEqual(initialClusterCount);
    }

    /**
     *
     * @param iterationCount
     * @return
     */
    public static FixedIterationCountCondition iterationCountGreaterThan(int iterationCount) {
        return new FixedIterationCountCondition(iterationCount);
    }

    /**
     *
     * @param iterationHistory
     * @return
     */
    public boolean isSatisfied(IterationHistory iterationHistory) {
        return iterationCountCondition.apply(iterationHistory == null ? 0 : iterationHistory.getIterationCount());
    }

}
