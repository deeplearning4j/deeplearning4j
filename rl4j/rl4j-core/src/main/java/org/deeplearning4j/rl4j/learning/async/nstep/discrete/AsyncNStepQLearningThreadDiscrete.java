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

package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.async.AsyncThreadDiscrete;
import org.deeplearning4j.rl4j.learning.async.IAsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.UpdateAlgorithm;
import org.deeplearning4j.rl4j.learning.configuration.AsyncQLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public class AsyncNStepQLearningThreadDiscrete<OBSERVATION extends Encodable> extends AsyncThreadDiscrete<OBSERVATION, IDQN> {

    @Getter
    final protected AsyncQLearningConfiguration configuration;
    @Getter
    final protected IAsyncGlobal<IDQN> asyncGlobal;
    @Getter
    final protected int threadNumber;

    final private Random rnd;

    public AsyncNStepQLearningThreadDiscrete(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IAsyncGlobal<IDQN> asyncGlobal,
                                             AsyncQLearningConfiguration configuration,
                                             TrainingListenerList listeners, int threadNumber, int deviceNum) {
        super(asyncGlobal, mdp, listeners, threadNumber, deviceNum);
        this.configuration = configuration;
        this.asyncGlobal = asyncGlobal;
        this.threadNumber = threadNumber;
        rnd = Nd4j.getRandom();

        Long seed = configuration.getSeed();
        if(seed != null) {
            rnd.setSeed(seed + threadNumber);
        }

        setUpdateAlgorithm(buildUpdateAlgorithm());
    }

    public Policy<Integer> getPolicy(IDQN nn) {
        return new EpsGreedy(new DQNPolicy(nn), getMdp(), configuration.getUpdateStart(), configuration.getEpsilonNbStep(),
                rnd, configuration.getMinEpsilon(), this);
    }

    @Override
    protected UpdateAlgorithm<IDQN> buildUpdateAlgorithm() {
        return new QLearningUpdateAlgorithm(getMdp().getActionSpace().getSize(), configuration.getGamma());
    }
}
