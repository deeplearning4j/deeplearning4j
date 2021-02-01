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

package org.deeplearning4j.eval.curves;

import org.nd4j.shade.guava.base.Preconditions;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * @deprecated Use {@link org.nd4j.evaluation.curves.ReliabilityDiagram}
 */
@Deprecated
@Data
@EqualsAndHashCode(callSuper = true)
public class PrecisionRecallCurve extends org.nd4j.evaluation.curves.PrecisionRecallCurve{

    /**
     * @deprecated Use {@link org.nd4j.evaluation.curves.ReliabilityDiagram}
     */
    @Deprecated
    public PrecisionRecallCurve(@JsonProperty("threshold") double[] threshold,
                    @JsonProperty("precision") double[] precision, @JsonProperty("recall") double[] recall,
                    @JsonProperty("tpCount") int[] tpCount, @JsonProperty("fpCount") int[] fpCount,
                    @JsonProperty("fnCount") int[] fnCount, @JsonProperty("totalCount") int totalCount) {
        super(threshold, precision, recall, tpCount, fpCount, fnCount, totalCount);
    }

    public static class Point extends org.nd4j.evaluation.curves.PrecisionRecallCurve.Point{
        public Point(int idx, double threshold, double precision, double recall) {
            super(idx, threshold, precision, recall);
        }
    }

    public static class Confusion extends org.nd4j.evaluation.curves.PrecisionRecallCurve.Confusion{
        public Confusion(org.nd4j.evaluation.curves.PrecisionRecallCurve.Point point, int tpCount, int fpCount, int fnCount, int tnCount) {
            super(point, tpCount, fpCount, fnCount, tnCount);
        }
    }
}
