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

package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/7/16.
 * Specialized constructors for the Conv (pixels input) case
 * Specialized conf + provide additional type safety
 */
public class AsyncNStepQLearningDiscreteConv<O extends Encodable> extends AsyncNStepQLearningDiscrete<O> {

    final private HistoryProcessor.Configuration hpconf;

    public AsyncNStepQLearningDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn,
                    HistoryProcessor.Configuration hpconf, AsyncNStepQLConfiguration conf, DataManager dataManager) {
        super(mdp, dqn, conf, dataManager);
        this.hpconf = hpconf;
        setHistoryProcessor(hpconf);
    }

    public AsyncNStepQLearningDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, DQNFactory factory,
                    HistoryProcessor.Configuration hpconf, AsyncNStepQLConfiguration conf, DataManager dataManager) {
        this(mdp, factory.buildDQN(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf, dataManager);
    }

    public AsyncNStepQLearningDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, DQNFactoryStdConv.Configuration netConf,
                    HistoryProcessor.Configuration hpconf, AsyncNStepQLConfiguration conf, DataManager dataManager) {
        this(mdp, new DQNFactoryStdConv(netConf), hpconf, conf, dataManager);
    }

    @Override
    public AsyncThread newThread(int i) {
        AsyncThread at = super.newThread(i);
        at.setHistoryProcessor(hpconf);
        return at;
    }
}
