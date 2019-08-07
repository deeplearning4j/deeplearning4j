/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManagerSyncTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 */
public class QLearningDiscreteDense<O extends Encodable> extends QLearningDiscrete<O> {


    @Deprecated
    public QLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLearning.QLConfiguration conf,
                    IDataManager dataManager) {
        this(mdp, dqn, conf);
        addListener(DataManagerSyncTrainingListener.builder(dataManager).build());
    }
    public QLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLearning.QLConfiguration conf) {
        super(mdp, dqn, conf, conf.getEpsilonNbStep());
    }

    @Deprecated
    public QLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, DQNFactory factory,
                    QLearning.QLConfiguration conf, IDataManager dataManager) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                        dataManager);
    }
    public QLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                  QLearning.QLConfiguration conf) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    @Deprecated
    public QLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, DQNFactoryStdDense.Configuration netConf,
                    QLearning.QLConfiguration conf, IDataManager dataManager) {
        this(mdp, new DQNFactoryStdDense(netConf), conf, dataManager);
    }
    public QLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, DQNFactoryStdDense.Configuration netConf,
                                  QLearning.QLConfiguration conf) {
        this(mdp, new DQNFactoryStdDense(netConf), conf);
    }

}
