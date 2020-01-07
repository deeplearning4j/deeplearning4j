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

package org.deeplearning4j.rl4j.learning.async;

import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Stack;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * Async Learning specialized for the Discrete Domain
 *
 */
public abstract class AsyncThreadDiscrete<O, NN extends NeuralNet>
                extends AsyncThread<O, Integer, DiscreteSpace, NN> {

    @Getter
    private NN current;

    public AsyncThreadDiscrete(IAsyncGlobal<NN> asyncGlobal, MDP<O, Integer, DiscreteSpace> mdp, TrainingListenerList listeners, int threadNumber, int deviceNum) {
        super(asyncGlobal, mdp, listeners, threadNumber, deviceNum);
        synchronized (asyncGlobal) {
            current = (NN)asyncGlobal.getCurrent().clone();
        }
    }

    /**
     * "Subepoch"  correspond to the t_max-step iterations
     * that stack rewards with t_max MiniTrans
     *
     * @param sObs the obs to start from
     * @param nstep the number of max nstep (step until t_max or state is terminal)
     * @return subepoch training informations
     */
    public SubEpochReturn trainSubEpoch(Observation sObs, int nstep) {

        synchronized (getAsyncGlobal()) {
            current.copy(getAsyncGlobal().getCurrent());
        }
        Stack<MiniTrans<Integer>> rewards = new Stack<>();

        Observation obs = sObs;
        IPolicy<O, Integer> policy = getPolicy(current);

        Integer action;
        Integer lastAction = getMdp().getActionSpace().noOp();
        IHistoryProcessor hp = getHistoryProcessor();
        int skipFrame = hp != null ? hp.getConf().getSkipFrame() : 1;

        double reward = 0;
        double accuReward = 0;
        int stepAtStart = getCurrentEpochStep();
        int lastStep = nstep * skipFrame + stepAtStart;
        while (!getMdp().isDone() && getCurrentEpochStep() < lastStep) {

            //if step of training, just repeat lastAction
            if (obs.isSkipped()) {
                action = lastAction;
            } else {
                action = policy.nextAction(obs);
            }

            StepReply<Observation> stepReply = getLegacyMDPWrapper().step(action);
            accuReward += stepReply.getReward() * getConf().getRewardFactor();

            //if it's not a skipped frame, you can do a step of training
            if (!obs.isSkipped() || stepReply.isDone()) {

                INDArray[] output = current.outputAll(obs.getData());
                rewards.add(new MiniTrans(obs.getData(), action, output, accuReward));

                accuReward = 0;
            }

            obs = stepReply.getObservation();

            reward += stepReply.getReward();

            incrementStep();
            lastAction = action;
        }

        //a bit of a trick usable because of how the stack is treated to init R
        // FIXME: The last element of minitrans is only used to seed the reward in calcGradient; observation, action and output are ignored.

        if (getMdp().isDone() && getCurrentEpochStep() < lastStep)
            rewards.add(new MiniTrans(obs.getData(), null, null, 0));
        else {
            INDArray[] output = null;
            if (getConf().getTargetDqnUpdateFreq() == -1)
                output = current.outputAll(obs.getData());
            else synchronized (getAsyncGlobal()) {
                output = getAsyncGlobal().getTarget().outputAll(obs.getData());
            }
            double maxQ = Nd4j.max(output[0]).getDouble(0);
            rewards.add(new MiniTrans(obs.getData(), null, output, maxQ));
        }

        getAsyncGlobal().enqueue(calcGradient(current, rewards), getCurrentEpochStep());

        return new SubEpochReturn(getCurrentEpochStep() - stepAtStart, obs, reward, current.getLatestScore());
    }

    public abstract Gradient[] calcGradient(NN nn, Stack<MiniTrans<Integer>> rewards);
}
