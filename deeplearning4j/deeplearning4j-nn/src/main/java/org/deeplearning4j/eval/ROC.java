/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.eval;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

/**
 * @deprecated Use {@link org.nd4j.evaluation.classification.ROC}
 */
@Deprecated
@EqualsAndHashCode(callSuper = true)
@Data
public class ROC extends org.nd4j.evaluation.classification.ROC implements IEvaluation<org.nd4j.evaluation.classification.ROC> {

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ROC}
     */
    @Deprecated
    public ROC() { }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ROC}
     */
    @Deprecated
    public ROC(int thresholdSteps) {
        super(thresholdSteps);
    }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ROC}
     */
    @Deprecated
    public ROC(int thresholdSteps, boolean rocRemoveRedundantPts) {
        super(thresholdSteps, rocRemoveRedundantPts);
    }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ROC}
     */
    @Deprecated
    public ROC(int thresholdSteps, boolean rocRemoveRedundantPts, int exactAllocBlockSize) {
        super(thresholdSteps, rocRemoveRedundantPts, exactAllocBlockSize);
    }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ROC.CountsForThreshold}
     */
    @Deprecated
    @NoArgsConstructor
    public static class CountsForThreshold extends org.nd4j.evaluation.classification.ROC.CountsForThreshold {

        public CountsForThreshold(double threshold) {
            super(threshold);
        }

        public CountsForThreshold(double threshold, long countTruePositive, long countFalsePositive){
            super(threshold, countTruePositive, countFalsePositive);
        }

        @Override
        public CountsForThreshold clone() {
            return new CountsForThreshold(getThreshold(), getCountTruePositive(), getCountFalsePositive());
        }
    }
}
