/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
 *
 *  This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.async.AsyncThreadDiscrete;
import org.deeplearning4j.rl4j.learning.async.IAsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.UpdateAlgorithm;
import org.deeplearning4j.rl4j.learning.configuration.AsyncQLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public class AsyncNStepQLearningThreadDiscrete<O extends Encodable> extends AsyncThreadDiscrete<O, IDQN> {

    @Getter
    final protected AsyncQLearningConfiguration conf;
    @Getter
    final protected IAsyncGlobal<IDQN> asyncGlobal;
    @Getter
    final protected int threadNumber;

    final private Random rnd;

    public AsyncNStepQLearningThreadDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IAsyncGlobal<IDQN> asyncGlobal,
                                             AsyncQLearningConfiguration conf,
                                             TrainingListenerList listeners, int threadNumber, int deviceNum) {
        super(asyncGlobal, mdp, listeners, threadNumber, deviceNum);
        this.conf = conf;
        this.asyncGlobal = asyncGlobal;
        this.threadNumber = threadNumber;
        rnd = Nd4j.getRandom();

        Long seed = conf.getSeed();
        if (seed != null) {
            rnd.setSeed(seed + threadNumber);
        }

        setUpdateAlgorithm(buildUpdateAlgorithm());
    }

    public Policy<O, Integer> getPolicy(IDQN nn) {
        return new EpsGreedy(new DQNPolicy(nn), getMdp(), conf.getUpdateStart(), conf.getEpsilonNbStep(),
                rnd, conf.getMinEpsilon(), this);
    }

    @Override
    protected UpdateAlgorithm<IDQN> buildUpdateAlgorithm() {
        int[] shape = getHistoryProcessor() == null ? getMdp().getObservationSpace().getShape() : getHistoryProcessor().getConf().getShape();
        return new QLearningUpdateAlgorithm(shape, getMdp().getActionSpace().getSize(), conf.getGamma());
    }
}
