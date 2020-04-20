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

package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.async.*;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.rng.Random;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 *
 * Local thread as described in the https://arxiv.org/abs/1602.01783 paper.
 */
public class A3CThreadDiscrete<O extends Encodable> extends AsyncThreadDiscrete<O, IActorCritic> {

    @Getter
    final protected A3CLearningConfiguration conf;
    @Getter
    final protected IAsyncGlobal<IActorCritic> asyncGlobal;
    @Getter
    final protected int threadNumber;

    final private Random rnd;

    public A3CThreadDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IAsyncGlobal<IActorCritic> asyncGlobal,
                             A3CLearningConfiguration a3cc, int deviceNum, TrainingListenerList listeners,
                             int threadNumber) {
        super(asyncGlobal, mdp, listeners, threadNumber, deviceNum);
        this.conf = a3cc;
        this.asyncGlobal = asyncGlobal;
        this.threadNumber = threadNumber;

        Long seed = conf.getSeed();
        rnd = Nd4j.getRandom();
        if(seed != null) {
            rnd.setSeed(seed + threadNumber);
        }

        setUpdateAlgorithm(buildUpdateAlgorithm());
    }

    @Override
    protected Policy<O, Integer> getPolicy(IActorCritic net) {
        return new ACPolicy(net, rnd);
    }

    @Override
    protected UpdateAlgorithm<IActorCritic> buildUpdateAlgorithm() {
        int[] shape = getHistoryProcessor() == null ? getMdp().getObservationSpace().getShape() : getHistoryProcessor().getConf().getShape();
        return new AdvantageActorCriticUpdateAlgorithm(asyncGlobal.getTarget().isRecurrent(), shape, getMdp().getActionSpace().getSize(), conf.getGamma());
    }
}
