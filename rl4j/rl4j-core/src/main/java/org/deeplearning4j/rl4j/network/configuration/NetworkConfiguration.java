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

package org.deeplearning4j.rl4j.network.configuration;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.Singular;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.List;


@Data
@SuperBuilder
@NoArgsConstructor
public class NetworkConfiguration {

    /**
     * The learning rate of the network
     */
    @Builder.Default
    private double learningRate = 0.01;

    /**
     * L2 regularization on the network
     */
    @Builder.Default
    private double l2 = 0.0;

    /**
     * The network's gradient update algorithm
     */
    private IUpdater updater;

    /**
     * Training listeners attached to the network
     */
    @Singular
    private List<TrainingListener> listeners;

}
