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

import lombok.EqualsAndHashCode;

/**
 * @deprecated Use {@link org.nd4j.evaluation.classification.ROCBinary}
 */
@Deprecated
@EqualsAndHashCode(callSuper = true)
public class ROCBinary extends org.nd4j.evaluation.classification.ROCBinary implements org.deeplearning4j.eval.IEvaluation<org.nd4j.evaluation.classification.ROCBinary> {
    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ROCBinary}
     */
    @Deprecated
    public static final int DEFAULT_STATS_PRECISION = 4;

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ROCBinary}
     */
    @Deprecated
    public ROCBinary() { }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ROCBinary}
     */
    @Deprecated
    public ROCBinary(int thresholdSteps) {
        super(thresholdSteps);
    }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ROCBinary}
     */
    @Deprecated
    public ROCBinary(int thresholdSteps, boolean rocRemoveRedundantPts) {
        super(thresholdSteps, rocRemoveRedundantPts);
    }
}
