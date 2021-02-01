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
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@EqualsAndHashCode(callSuper = true)
public class A3CLearningConfiguration extends LearningConfiguration implements IAsyncLearningConfiguration {

    /**
     * The number of asynchronous threads to use to generate gradients
     */
    private final int numThreads;

    /**
     * The number of steps to calculate gradients over
     */
    private final int nStep;

    /**
     * The frequency of async training iterations to update the target network.
     *
     * If this is set to -1 then the target network is updated after every training iteration
     */
    @Builder.Default
    private int learnerUpdateFrequency = -1;
}
