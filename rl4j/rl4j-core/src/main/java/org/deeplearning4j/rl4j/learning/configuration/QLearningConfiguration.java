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

package org.deeplearning4j.rl4j.learning.configuration;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class QLearningConfiguration extends LearningConfiguration {

    /**
     * The maximum size of the experience replay buffer
     */
    @Builder.Default
    private int expRepMaxSize = 150000;

    /**
     * The batch size of experience for each training iteration
     */
    @Builder.Default
    private int batchSize = 32;

    /**
     * How many steps between target network updates
     */
    @Builder.Default
    private int targetDqnUpdateFreq = 100;

    /**
     * The number of steps to initially wait for until samplling batches from experience replay buffer
     */
    @Builder.Default
    private int updateStart = 10;

    /**
     * Prevent the new Q-Value from being farther than <i>errorClamp</i> away from the previous value. Double.NaN will result in no clamping
     */
    @Builder.Default
    private double errorClamp = 1.0;

    /**
     * The minimum probability for random exploration action during episilon-greedy annealing
     */
    @Builder.Default
    private double minEpsilon = 0.1f;

    /**
     * The number of steps to anneal epsilon to its minimum value.
     */
    @Builder.Default
    private int epsilonNbStep = 10000;

    /**
     * Whether to use the double DQN algorithm
     */
    @Builder.Default
    private boolean doubleDQN = false;

}
