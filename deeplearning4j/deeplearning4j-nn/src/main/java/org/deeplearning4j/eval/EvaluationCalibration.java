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

package org.deeplearning4j.eval;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * @deprecated Use {@link org.nd4j.evaluation.classification.EvaluationCalibration}
 */
@Deprecated
@Getter
@EqualsAndHashCode
public class EvaluationCalibration extends org.nd4j.evaluation.classification.EvaluationCalibration implements org.deeplearning4j.eval.IEvaluation<org.nd4j.evaluation.classification.EvaluationCalibration> {

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.EvaluationCalibration}
     */
    @Deprecated
    public EvaluationCalibration() {
        super();
    }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.EvaluationCalibration}
     */
    @Deprecated
    public EvaluationCalibration(int reliabilityDiagNumBins, int histogramNumBins) {
        super(reliabilityDiagNumBins, histogramNumBins);
    }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.EvaluationCalibration}
     */
    @Deprecated
    public EvaluationCalibration(@JsonProperty("reliabilityDiagNumBins") int reliabilityDiagNumBins,
                    @JsonProperty("histogramNumBins") int histogramNumBins,
                    @JsonProperty("excludeEmptyBins") boolean excludeEmptyBins) {
        super(reliabilityDiagNumBins, histogramNumBins, excludeEmptyBins);
    }
}
