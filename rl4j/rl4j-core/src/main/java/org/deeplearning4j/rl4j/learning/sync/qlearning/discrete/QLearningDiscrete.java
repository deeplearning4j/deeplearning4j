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

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.TDTargetAlgorithm.*;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 *
 * DQN or Deep Q-Learning in the Discrete domain
 *
 * http://arxiv.org/abs/1312.5602
 *
 */
public abstract class QLearningDiscrete<O extends Encodable> extends QLearning<O, Integer, DiscreteSpace> {

    @Getter
    final private QLConfiguration configuration;
    private final LegacyMDPWrapper<O, Integer, DiscreteSpace> mdp;
    @Getter
    private DQNPolicy<O> policy;
    @Getter
    private EpsGreedy<O, Integer, DiscreteSpace> egPolicy;

    @Getter
    final private IDQN qNetwork;
    @Getter
    @Setter(AccessLevel.PROTECTED)
    private IDQN targetQNetwork;

    private int lastAction;
    private INDArray[] history = null;
    private double accuReward = 0;

    ITDTargetAlgorithm tdTargetAlgorithm;

    public QLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLConfiguration conf,
                             int epsilonNbStep) {
        this(mdp, dqn, conf, epsilonNbStep, Nd4j.getRandomFactory().getNewRandomInstance(conf.getSeed()));
    }

    public QLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLConfiguration conf,
                             int epsilonNbStep, Random random) {
        super(conf);
        this.configuration = conf;
        this.mdp = new LegacyMDPWrapper<O, Integer, DiscreteSpace>(mdp);
        qNetwork = dqn;
        targetQNetwork = dqn.clone();
        policy = new DQNPolicy(getQNetwork());
        egPolicy = new EpsGreedy(policy, mdp, conf.getUpdateStart(), epsilonNbStep, random, conf.getMinEpsilon(),
                this);

        tdTargetAlgorithm = conf.isDoubleDQN()
                ? new DoubleDQN(this, conf.getGamma(), conf.getErrorClamp())
                : new StandardDQN(this, conf.getGamma(), conf.getErrorClamp());

    }

    public MDP<O, Integer, DiscreteSpace> getMdp() {
        return mdp.getWrappedMDP();
    }

    public void postEpoch() {

        if (getHistoryProcessor() != null)
            getHistoryProcessor().stopMonitor();

    }

    public void preEpoch() {
        history = null;
        lastAction = 0;
        accuReward = 0;
    }

    /**
     * Single step of training
     * @param obs last obs
     * @return relevant info for next step
     */
    protected QLStepReturn<Observation> trainStep(Observation obs) {

        Integer action;
        INDArray input = obs.toINDArray();
        boolean isHistoryProcessor = getHistoryProcessor() != null;


        int skipFrame = isHistoryProcessor ? getHistoryProcessor().getConf().getSkipFrame() : 1;
        int historyLength = isHistoryProcessor ? getHistoryProcessor().getConf().getHistoryLength() : 1;
        int updateStart = getConfiguration().getUpdateStart()
                        + ((getConfiguration().getBatchSize() + historyLength) * skipFrame);

        Double maxQ = Double.NaN; //ignore if Nan for stats

        //if step of training, just repeat lastAction
        if (getStepCounter() % skipFrame != 0) {
            action = lastAction;
        } else {
            if (history == null) {
                if (isHistoryProcessor) {
                    getHistoryProcessor().add(input);
                    history = getHistoryProcessor().getHistory();
                } else
                    history = new INDArray[] {input};
            }
            //concat the history into a single INDArray input
            INDArray hstack = Transition.concat(Transition.dup(history));
            if (isHistoryProcessor) {
                hstack.muli(1.0 / getHistoryProcessor().getScale());
            }

            //if input is not 2d, you have to append that the batch is 1 length high
            if (hstack.shape().length > 2)
                hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape())));

            INDArray qs = getQNetwork().output(hstack);
            int maxAction = Learning.getMaxAction(qs);

            maxQ = qs.getDouble(maxAction);
            action = getEgPolicy().nextAction(hstack);
        }

        lastAction = action;

        StepReply<Observation> stepReply = mdp.step(action);

        INDArray ninput = stepReply.getObservation().toINDArray();

        if (isHistoryProcessor)
            getHistoryProcessor().record(ninput);

        accuReward += stepReply.getReward() * configuration.getRewardFactor();

        //if it's not a skipped frame, you can do a step of training
        if (getStepCounter() % skipFrame == 0 || stepReply.isDone()) {

            if (isHistoryProcessor)
                getHistoryProcessor().add(ninput);

            INDArray[] nhistory = isHistoryProcessor ? getHistoryProcessor().getHistory() : new INDArray[] {ninput};

            Transition<Integer> trans = new Transition(history, action, accuReward, stepReply.isDone(), nhistory[0]);
            getExpReplay().store(trans);

            if (getStepCounter() > updateStart) {
                DataSet targets = setTarget(getExpReplay().getBatch());
                getQNetwork().fit(targets.getFeatures(), targets.getLabels());
            }

            history = nhistory;
            accuReward = 0;
        }

        return new QLStepReturn<Observation>(maxQ, getQNetwork().getLatestScore(), stepReply);
    }

    protected DataSet setTarget(ArrayList<Transition<Integer>> transitions) {
        if (transitions.size() == 0)
            throw new IllegalArgumentException("too few transitions");

        // TODO: Remove once we use DataSets in observations
        int[] shape = getHistoryProcessor() == null ? getMdp().getObservationSpace().getShape()
                : getHistoryProcessor().getConf().getShape();
        ((BaseTDTargetAlgorithm) tdTargetAlgorithm).setNShape(makeShape(transitions.size(), shape));

        // TODO: Remove once we use DataSets in observations
        if(getHistoryProcessor() != null) {
            ((BaseTDTargetAlgorithm) tdTargetAlgorithm).setScale(getHistoryProcessor().getScale());
        }

        return tdTargetAlgorithm.computeTDTargets(transitions);
    }
}
