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

package org.nd4j.autodiff.samediff.training;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;

/**
 * A single (prompt, chosen, rejected) preference example for offline RL alignment.
 *
 * <p>Used by {@link RLAlignmentPipeline#trainDPO(java.util.List)} (and the SimPO,
 * KTO, ORPO equivalents) to describe one comparison between a preferred and a
 * dis-preferred completion for the same prompt.
 *
 * <p>Example:
 * <pre>
 * PreferencePair pair = PreferencePair.builder()
 *     .prompt("Explain quantum entanglement.")
 *     .chosen("Quantum entanglement is a phenomenon ...")
 *     .rejected("I don't know.")
 *     .build();
 * </pre>
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class PreferencePair {

    /**
     * The prompt shown to the model (user turn or instruction).
     */
    @NonNull
    private final String prompt;

    /**
     * The preferred (chosen / better) completion.
     */
    @NonNull
    private final String chosen;

    /**
     * The dis-preferred (rejected / worse) completion.
     */
    @NonNull
    private final String rejected;
}
