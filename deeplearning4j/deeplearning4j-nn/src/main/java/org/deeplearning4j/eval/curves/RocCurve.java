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

package org.deeplearning4j.eval.curves;

import org.nd4j.shade.guava.base.Preconditions;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * @deprecated Use {@link org.nd4j.evaluation.curves.RocCurve}
 */
@Deprecated
@Data
@EqualsAndHashCode(exclude = {"auc"}, callSuper = false)
public class RocCurve extends org.nd4j.evaluation.curves.RocCurve {

    /**
     * @deprecated Use {@link org.nd4j.evaluation.curves.RocCurve}
     */
    @Deprecated
    public RocCurve(@JsonProperty("threshold") double[] threshold, @JsonProperty("fpr") double[] fpr,
                    @JsonProperty("tpr") double[] tpr) {
        super(threshold, fpr, tpr);
    }


    /**
     * @deprecated Use {@link org.nd4j.evaluation.curves.RocCurve}
     */
    @Deprecated
    public static RocCurve fromJson(String json) {
        return fromJson(json, RocCurve.class);
    }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.curves.RocCurve}
     */
    @Deprecated
    public static RocCurve fromYaml(String yaml) {
        return fromYaml(yaml, RocCurve.class);
    }

}
