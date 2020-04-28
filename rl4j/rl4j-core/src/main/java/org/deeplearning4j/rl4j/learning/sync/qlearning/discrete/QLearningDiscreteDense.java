/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
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
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 */
public class QLearningDiscreteDense<OBSERVATION extends Encodable> extends QLearningDiscrete<OBSERVATION> {


    @Deprecated
    public QLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IDQN dqn, QLearning.QLConfiguration conf,
                                  IDataManager dataManager) {
        this(mdp, dqn, conf);
        addListener(new DataManagerTrainingListener(dataManager));
    }

    @Deprecated
    public QLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IDQN dqn, QLearning.QLConfiguration conf) {
        super(mdp, dqn, conf.toLearningConfiguration(), conf.getEpsilonNbStep());
    }

    public QLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IDQN dqn, QLearningConfiguration conf) {
        super(mdp, dqn, conf, conf.getEpsilonNbStep());
    }

    @Deprecated
    public QLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                  QLearning.QLConfiguration conf, IDataManager dataManager) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                        dataManager);
    }

    @Deprecated
    public QLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                  QLearning.QLConfiguration conf) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    public QLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                  QLearningConfiguration conf) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    @Deprecated
    public QLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactoryStdDense.Configuration netConf,
                                  QLearning.QLConfiguration conf, IDataManager dataManager) {

        this(mdp, new DQNFactoryStdDense(netConf.toNetworkConfiguration()), conf, dataManager);
    }

    @Deprecated
    public QLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactoryStdDense.Configuration netConf,
                                  QLearning.QLConfiguration conf) {
        this(mdp, new DQNFactoryStdDense(netConf.toNetworkConfiguration()), conf);
    }

    public QLearningDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNDenseNetworkConfiguration netConf,
                                  QLearningConfiguration conf) {
        this(mdp, new DQNFactoryStdDense(netConf), conf);
    }

}
