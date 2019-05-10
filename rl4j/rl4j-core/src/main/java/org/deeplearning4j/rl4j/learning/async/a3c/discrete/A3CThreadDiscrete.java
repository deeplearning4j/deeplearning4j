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

package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import lombok.Getter;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncThreadDiscrete;
import org.deeplearning4j.rl4j.learning.async.MiniTrans;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Random;
import java.util.Stack;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 *
 * Local thread as described in the https://arxiv.org/abs/1602.01783 paper.
 */
public class A3CThreadDiscrete<O extends Encodable> extends AsyncThreadDiscrete<O, IActorCritic> {

    @Getter
    final protected A3CDiscrete.A3CConfiguration conf;
    @Getter
    final protected MDP<O, Integer, DiscreteSpace> mdp;
    @Getter
    final protected AsyncGlobal<IActorCritic> asyncGlobal;
    @Getter
    final protected int threadNumber;
    @Getter
    final protected DataManager dataManager;

    final private Random random;

    public A3CThreadDiscrete(MDP<O, Integer, DiscreteSpace> mdp, AsyncGlobal<IActorCritic> asyncGlobal,
                    A3CDiscrete.A3CConfiguration a3cc, int threadNumber, DataManager dataManager) {
        super(asyncGlobal, threadNumber);
        this.conf = a3cc;
        this.asyncGlobal = asyncGlobal;
        this.threadNumber = threadNumber;
        this.mdp = mdp;
        this.dataManager = dataManager;
        mdp.getActionSpace().setSeed(conf.getSeed() + threadNumber);
        random = new Random(conf.getSeed() + threadNumber);
    }

    @Override
    protected Policy<O, Integer> getPolicy(IActorCritic net) {
        return new ACPolicy(net, random);
    }

    /**
     *  calc the gradients based on the n-step rewards
     */
    @Override
    public Gradient[] calcGradient(IActorCritic iac, Stack<MiniTrans<Integer>> rewards) {
        MiniTrans<Integer> minTrans = rewards.pop();

        int size = rewards.size();

        //if recurrent then train as a time serie with a batch size of 1
        boolean recurrent = getAsyncGlobal().getCurrent().isRecurrent();

        int[] shape = getHistoryProcessor() == null ? mdp.getObservationSpace().getShape()
                        : getHistoryProcessor().getConf().getShape();
        int[] nshape = recurrent ? Learning.makeShape(1, shape, size)
                        : Learning.makeShape(size, shape);

        INDArray input = Nd4j.create(nshape);
        INDArray targets = recurrent ? Nd4j.create(1, 1, size) : Nd4j.create(size, 1);
        INDArray logSoftmax = recurrent ? Nd4j.zeros(1, mdp.getActionSpace().getSize(), size)
                        : Nd4j.zeros(size, mdp.getActionSpace().getSize());

        double r = minTrans.getReward();
        for (int i = size - 1; i >= 0; i--) {
            minTrans = rewards.pop();

            r = minTrans.getReward() + conf.getGamma() * r;
            if (recurrent) {
                input.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i)).assign(minTrans.getObs());
            } else {
                input.putRow(i, minTrans.getObs());
            }

            //the critic
            targets.putScalar(i, r);

            //the actor
            double expectedV = minTrans.getOutput()[0].getDouble(0);
            double advantage = r - expectedV;
            if (recurrent) {
                logSoftmax.putScalar(0, minTrans.getAction(), i, advantage);
            } else {
                logSoftmax.putScalar(i, minTrans.getAction(), advantage);
            }
        }

        return iac.gradient(input, new INDArray[] {targets, logSoftmax});
    }
}
