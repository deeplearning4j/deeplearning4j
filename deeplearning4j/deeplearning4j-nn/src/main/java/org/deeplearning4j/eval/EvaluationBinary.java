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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Use {@link org.nd4j.evaluation.classification.EvaluationBinary}
 */
@Deprecated
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
@Data
public class EvaluationBinary extends org.nd4j.evaluation.classification.EvaluationBinary implements org.deeplearning4j.eval.IEvaluation<org.nd4j.evaluation.classification.EvaluationBinary> {
    @Deprecated
    public static final int DEFAULT_PRECISION = 4;
    @Deprecated
    public static final double DEFAULT_EDGE_VALUE = 0.0;

    /**
     * Use {@link org.nd4j.evaluation.classification.EvaluationBinary}
     */
    @Deprecated
    public EvaluationBinary(INDArray decisionThreshold) {
        super(decisionThreshold);
    }

    /**
     * Use {@link org.nd4j.evaluation.classification.EvaluationBinary}
     */
    @Deprecated
    public EvaluationBinary(int size, Integer rocBinarySteps) {
        super(size, rocBinarySteps);
    }

    /**
     * Use {@link org.nd4j.evaluation.classification.EvaluationBinary#fromJson(String)}
     */
    @Deprecated
    public static EvaluationBinary fromJson(String json) {
        return fromJson(json, EvaluationBinary.class);
    }

    /**
     * Use {@link org.nd4j.evaluation.classification.EvaluationBinary.fromYaml(String)}
     */
    @Deprecated
    public static EvaluationBinary fromYaml(String yaml) {
        return fromYaml(yaml, EvaluationBinary.class);
    }


}
