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

package org.deeplearning4j.rl4j.experience;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.observation.IObservationSource;
import org.deeplearning4j.rl4j.observation.Observation;

@Data
public class StateActionRewardState<A> implements IObservationSource {

    @Getter
    Observation observation;

    A action;
    double reward;
    boolean isTerminal;

    @Getter @Setter
    Observation nextObservation;

    public StateActionRewardState(Observation observation, A action, double reward, boolean isTerminal) {
        this.observation = observation;
        this.action = action;
        this.reward = reward;
        this.isTerminal = isTerminal;
        this.nextObservation = null;
    }

    private StateActionRewardState(Observation observation, A action, double reward, boolean isTerminal, Observation nextObservation) {
        this.observation = observation;
        this.action = action;
        this.reward = reward;
        this.isTerminal = isTerminal;
        this.nextObservation = nextObservation;
    }

    /**
     * @return a duplicate of this instance
     */
    public StateActionRewardState<A> dup() {
        Observation dupObservation = observation.dup();
        Observation nextObs = nextObservation.dup();

        return new StateActionRewardState<A>(dupObservation, action, reward, isTerminal, nextObs);
    }
}
