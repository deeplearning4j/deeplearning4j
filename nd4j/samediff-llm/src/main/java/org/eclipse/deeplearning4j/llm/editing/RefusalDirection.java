/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.editing;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Represents a refusal direction found in a specific layer and position
 * of a language model's residual stream.
 *
 * <p>The direction is a unit vector in the model's hidden dimension space.
 * The score indicates how strongly this direction separates harmful from
 * harmless activations — higher scores indicate stronger refusal encoding.</p>
 */
@Data
@AllArgsConstructor
public class RefusalDirection {

    /** Name of the variable where this direction was found. */
    private final String layerName;

    /** Position in the residual stream where the direction was measured. */
    private final AbliterationConfig.ResidualStreamPosition position;

    /** Unit direction vector (shape: [hiddenDim]). */
    private final INDArray direction;

    /** Ranking score — magnitude of mean projection onto this direction. */
    private final double score;

    /** Layer index in the model (0-based). */
    private final int layerIndex;

    @Override
    public String toString() {
        return String.format("RefusalDirection{layer=%d (%s), position=%s, score=%.6f}",
                layerIndex, layerName, position, score);
    }
}
