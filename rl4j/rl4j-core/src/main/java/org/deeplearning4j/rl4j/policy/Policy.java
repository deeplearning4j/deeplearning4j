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

package org.deeplearning4j.rl4j.policy;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.EpochStepCounter;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 *
 * Abstract class common to all policies
 *
 * A Policy responsability is to choose the next action given a state
 */
public abstract class Policy<O, A> implements IPolicy<O, A> {

    public abstract NeuralNet getNeuralNet();

    public abstract A nextAction(Observation obs);

    public <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp) {
        return play(mdp, (IHistoryProcessor)null);
    }

    public <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp, HistoryProcessor.Configuration conf) {
        return play(mdp, new HistoryProcessor(conf));
    }

    @Override
    public <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp, IHistoryProcessor hp) {
        resetNetworks();

        RefacEpochStepCounter epochStepCounter = new RefacEpochStepCounter();
        LegacyMDPWrapper<O, A, AS> mdpWrapper = new LegacyMDPWrapper<O, A, AS>(mdp, hp, epochStepCounter);

        Learning.InitMdp<Observation> initMdp = refacInitMdp(mdpWrapper, hp, epochStepCounter);
        Observation obs = initMdp.getLastObs();

        double reward = initMdp.getReward();

        A lastAction = mdpWrapper.getActionSpace().noOp();
        A action;

        while (!mdpWrapper.isDone()) {

            if (obs.isSkipped()) {
                action = lastAction;
            } else {
                action = nextAction(obs);
            }

            lastAction = action;

            StepReply<Observation> stepReply = mdpWrapper.step(action);
            reward += stepReply.getReward();

            obs = stepReply.getObservation();
            epochStepCounter.incrementEpochStep();
        }

        return reward;
    }

    protected void resetNetworks() {
        getNeuralNet().reset();
    }

    protected <AS extends ActionSpace<A>> Learning.InitMdp<Observation> refacInitMdp(LegacyMDPWrapper<O, A, AS> mdpWrapper, IHistoryProcessor hp, RefacEpochStepCounter epochStepCounter) {
        epochStepCounter.setCurrentEpochStep(0);

        double reward = 0;

        Observation observation = mdpWrapper.reset();

        A action = mdpWrapper.getActionSpace().noOp(); //by convention should be the NO_OP
        while (observation.isSkipped() && !mdpWrapper.isDone()) {

            StepReply<Observation> stepReply = mdpWrapper.step(action);

            reward += stepReply.getReward();
            observation = stepReply.getObservation();

            epochStepCounter.incrementEpochStep();
        }

        return new Learning.InitMdp(0, observation, reward);
    }

    public class RefacEpochStepCounter implements EpochStepCounter {

        @Getter
        @Setter
        private int currentEpochStep = 0;

        public void incrementEpochStep() {
            ++currentEpochStep;
        }

    }
}
