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

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Result of an abliteration operation, containing the directions found,
 * which were applied, and what weights were modified.
 */
@Data
@Builder
public class AbliterationResult {

    /** All candidate refusal directions found, ranked by score descending. */
    private final List<RefusalDirection> allCandidates;

    /** The direction(s) that were actually applied to the weights. */
    private final List<RefusalDirection> appliedDirections;

    /** Total number of weight matrices that were modified. */
    private final int numWeightsModified;

    /** Names of weight variables that were modified. */
    private final List<String> modifiedWeightNames;

    @Override
    public String toString() {
        return String.format("AbliterationResult{candidates=%d, applied=%d, weightsModified=%d}",
                allCandidates.size(), appliedDirections.size(), numWeightsModified);
    }
}
