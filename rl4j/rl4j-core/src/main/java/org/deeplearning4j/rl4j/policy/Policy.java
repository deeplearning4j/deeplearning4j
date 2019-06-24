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

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearningInitializer;
import org.deeplearning4j.rl4j.learning.LearningInitializer;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.deeplearning4j.rl4j.observation.transforms.ObservationTransform;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 *
 * Abstract class common to all policies
 *
 * A Policy responsability is to choose the next action given a state
 */
public abstract class Policy<O extends Observation, A> {

    public abstract NeuralNet getNeuralNet();

    public abstract A nextAction(INDArray input);

    public <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp) {
        return play(mdp, null, null);
    }

    public <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp, HistoryProcessor.Configuration conf) {
        return play(mdp, new HistoryProcessor(conf), null);
    }

    public <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp, IHistoryProcessor hp) {
        return play(mdp, hp, null);
    }

    public <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp, IHistoryProcessor hp, ObservationTransform transform) {
        getNeuralNet().reset();

        ILearningInitializer<O, A, AS> initializer = new LearningInitializer<O, A, AS>();

        Learning.InitMdp<O> initMdp = initializer.initMdp(mdp, transform);

        Observation obs = initMdp.getLastObs();

        double reward = initMdp.getReward();

        A lastAction = mdp.getActionSpace().noOp();
        A action;
        INDArray history = null;

        while (!mdp.isDone()) {

            boolean isHistoryProcessor = hp != null;

            if(obs instanceof VoidObservation) {
                action = lastAction;
            } else {

                if (history == null) {
                    if (isHistoryProcessor) {
                        history = Transition.concat(hp.getHistory());
                    } else
                        history = obs.toNDArray();
                }
                INDArray hstack = history;
                if (isHistoryProcessor) {
                    hstack.muli(1.0 / hp.getScale());
                }
                if (getNeuralNet().isRecurrent()) {
                    //flatten everything for the RNN
                    hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape()), 1));
                } else {
                    if (hstack.shape().length > 2)
                        hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape())));
                }
                action = nextAction(hstack);
            }
            lastAction = action;

            StepReply<O> stepReply = mdp.step(action);
            obs = transform.getObservation(stepReply.getObservation());
            reward += stepReply.getReward();

            history = isHistoryProcessor
                    ? Transition.concat(hp.getHistory())
                    : obs.toNDArray();
        }


        return reward;
    }

}
