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
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@NoArgsConstructor
public class LearningConfiguration implements ILearningConfiguration {

    /**
     * Seed value used for training
     */
    @Builder.Default
    private Long seed = System.currentTimeMillis();

    /**
     * The maximum number of steps in each episode
     */
    @Builder.Default
    private int maxEpochStep = 200;

    /**
     * The maximum number of steps to train for
     */
    @Builder.Default
    private int maxStep = 150000;

    /**
     * Gamma parameter used for discounted rewards
     */
    @Builder.Default
    private double gamma = 0.99;

    /**
     * Scaling parameter for rewards
     */
    @Builder.Default
    private double rewardFactor = 1.0;

}
