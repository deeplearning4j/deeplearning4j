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

import org.deeplearning4j.rl4j.learning.configuration.AsyncQLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/7/16.
 */
public class AsyncNStepQLearningDiscreteDense<OBSERVATION extends Encodable> extends AsyncNStepQLearningDiscrete<OBSERVATION> {

    @Deprecated
    public AsyncNStepQLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IDQN dqn,
                                            AsyncNStepQLConfiguration conf, IDataManager dataManager) {
        super(mdp, dqn, conf.toLearningConfiguration());
        addListener(new DataManagerTrainingListener(dataManager));
    }

    @Deprecated
    public AsyncNStepQLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IDQN dqn,
                                            AsyncNStepQLConfiguration conf) {
        super(mdp, dqn, conf.toLearningConfiguration());
    }

    public AsyncNStepQLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IDQN dqn,
                                            AsyncQLearningConfiguration conf) {
        super(mdp, dqn, conf);
    }

    @Deprecated
    public AsyncNStepQLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                            AsyncNStepQLConfiguration conf, IDataManager dataManager) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                dataManager);
    }

    @Deprecated
    public AsyncNStepQLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                            AsyncNStepQLConfiguration conf) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    public AsyncNStepQLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                            AsyncQLearningConfiguration conf) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    @Deprecated
    public AsyncNStepQLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp,
                                            DQNFactoryStdDense.Configuration netConf, AsyncNStepQLConfiguration conf, IDataManager dataManager) {
        this(mdp, new DQNFactoryStdDense(netConf.toNetworkConfiguration()), conf, dataManager);
    }

    @Deprecated
    public AsyncNStepQLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp,
                                            DQNFactoryStdDense.Configuration netConf, AsyncNStepQLConfiguration conf) {
        this(mdp, new DQNFactoryStdDense(netConf.toNetworkConfiguration()), conf);
    }

    public AsyncNStepQLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp,
                                            DQNDenseNetworkConfiguration netConf, AsyncQLearningConfiguration conf) {
        this(mdp, new DQNFactoryStdDense(netConf), conf);
    }


}
