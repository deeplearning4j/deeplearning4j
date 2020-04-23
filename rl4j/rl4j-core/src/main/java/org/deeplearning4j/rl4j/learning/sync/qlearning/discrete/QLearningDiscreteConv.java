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

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.configuration.NetworkConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 * Specialized constructors for the Conv (pixels input) case
 * Specialized conf + provide additional type safety
 */
public class QLearningDiscreteConv<OBSERVATION extends Encodable> extends QLearningDiscrete<OBSERVATION> {


    @Deprecated
    public QLearningDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IDQN dqn, HistoryProcessor.Configuration hpconf,
                                 QLConfiguration conf, IDataManager dataManager) {
        this(mdp, dqn, hpconf, conf);
        addListener(new DataManagerTrainingListener(dataManager));
    }

    @Deprecated
    public QLearningDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IDQN dqn, HistoryProcessor.Configuration hpconf,
                                 QLConfiguration conf) {
        super(mdp, dqn, conf.toLearningConfiguration(), conf.getEpsilonNbStep() * hpconf.getSkipFrame());
        setHistoryProcessor(hpconf);
    }

    public QLearningDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IDQN dqn, HistoryProcessor.Configuration hpconf,
                                 QLearningConfiguration conf) {
        super(mdp, dqn, conf, conf.getEpsilonNbStep() * hpconf.getSkipFrame());
        setHistoryProcessor(hpconf);
    }

    @Deprecated
    public QLearningDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                 HistoryProcessor.Configuration hpconf, QLConfiguration conf, IDataManager dataManager) {
        this(mdp, factory.buildDQN(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf, dataManager);
    }

    @Deprecated
    public QLearningDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                 HistoryProcessor.Configuration hpconf, QLConfiguration conf) {
        this(mdp, factory.buildDQN(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf);
    }

    public QLearningDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                 HistoryProcessor.Configuration hpconf, QLearningConfiguration conf) {
        this(mdp, factory.buildDQN(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf);
    }

    @Deprecated
    public QLearningDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactoryStdConv.Configuration netConf,
                                 HistoryProcessor.Configuration hpconf, QLConfiguration conf, IDataManager dataManager) {
        this(mdp, new DQNFactoryStdConv(netConf.toNetworkConfiguration()), hpconf, conf, dataManager);
    }

    @Deprecated
    public QLearningDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactoryStdConv.Configuration netConf,
                                 HistoryProcessor.Configuration hpconf, QLConfiguration conf) {
        this(mdp, new DQNFactoryStdConv(netConf.toNetworkConfiguration()), hpconf, conf);
    }

    public QLearningDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, NetworkConfiguration netConf,
                                 HistoryProcessor.Configuration hpconf, QLearningConfiguration conf) {
        this(mdp, new DQNFactoryStdConv(netConf), hpconf, conf);
    }

}
