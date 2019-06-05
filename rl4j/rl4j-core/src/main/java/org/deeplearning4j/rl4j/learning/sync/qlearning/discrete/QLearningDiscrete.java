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

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.*;
import org.deeplearning4j.rl4j.mdp.HistoryProcessorMDPRunner;
import org.deeplearning4j.rl4j.mdp.IMDPRunner;
import org.deeplearning4j.rl4j.mdp.MDPRunner;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;
import org.deeplearning4j.rl4j.mdp.BaseMDPRunner;

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
    @Getter
    final private DataManager dataManager;
    @Getter
    final private MDP<O, Integer, DiscreteSpace> mdp;
    @Getter
    final private IDQN currentDQN;
    @Getter
    private DQNPolicy<O> policy;
    @Getter
    @Setter // FIXME: Remove, was added to help unit tests
    private EpsGreedy<O, Integer, DiscreteSpace> egPolicy;
    @Getter
    @Setter
    private IDQN targetDQN;

    @Getter @Setter // FIXME: Remove, was added to help unit tests
    private int lastAction;

    private double accuReward = 0;
    private int lastMonitor = -Constants.MONITOR_FREQ;

    @Getter // FIXME: Remove, was added to help unit tests
    private BaseMDPRunner<O, Integer> mdpRunner; // FIXME: Change type to IMDPRunner
    //private IMDPRunner mdpRunner; // FIXME: Temporary refac code

    public QLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLConfiguration conf,
                             DataManager dataManager, int epsilonNbStep) {
        super(conf);
        this.configuration = conf;
        this.mdp = mdp;
        this.dataManager = dataManager;
        currentDQN = dqn;
        targetDQN = dqn.clone();
        policy = new DQNPolicy(getCurrentDQN());
        egPolicy = new EpsGreedy(policy, mdp, conf.getUpdateStart(), epsilonNbStep, getRandom(), conf.getMinEpsilon(),
                this);
        mdp.getActionSpace().setSeed(conf.getSeed());
    }


    public void postEpoch() {

        if (getHistoryProcessor() != null)
            getHistoryProcessor().stopMonitor();

    }

    public void preEpoch() {
        // FIXME: Temporary refac code ---
        // FIXME: Find a way to set the HistoryProcessor in the ctor
        if(getHistoryProcessor() != null) {
            mdpRunner = new HistoryProcessorMDPRunner(getHistoryProcessor());
        } else {
            mdpRunner = new MDPRunner();
        }
        // ---

        mdpRunner.onPreEpoch();

        lastAction = 0;
        accuReward = 0;

        if (getStepCounter() - lastMonitor >= Constants.MONITOR_FREQ && getHistoryProcessor() != null
                && getDataManager().isSaveData()) {
            lastMonitor = getStepCounter();
            int[] shape = getMdp().getObservationSpace().getShape();
            getHistoryProcessor().startMonitor(getDataManager().getVideoDir() + "/video-" + getEpochCounter() + "-"
                    + getStepCounter() + ".mp4", shape);
        }
    }

    /**
     * Single step of training
     * @param obs last obs
     * @return relevant info for next step
     */
    protected QLStepReturn<O> trainStep(O obs) {

        Integer action;
        INDArray input = getInput(obs);

        boolean isHistoryProcessor = getHistoryProcessor() != null;


        if (isHistoryProcessor)
            getHistoryProcessor().record(input);

        mdpRunner.setStep(getStepCounter());
        mdpRunner.setMaxQ(Double.NaN); //ignore if Nan for stats

        action = mdpRunner.getNextAction(getCurrentDQN(), getEgPolicy(), input);
        action = action == null ? lastAction : action;

        lastAction = action;

        StepReply<O> stepReply = getMdp().step(action);

        accuReward += stepReply.getReward() * configuration.getRewardFactor();

        int skipFrame = isHistoryProcessor ? getHistoryProcessor().getConf().getSkipFrame() : 1;
        //if it's not a skipped frame, you can do a step of training
        if (getStepCounter() % skipFrame == 0 || stepReply.isDone()) {

            INDArray ninput = getInput(stepReply.getObservation());
            if (isHistoryProcessor)
                getHistoryProcessor().add(ninput);

            INDArray[] nhistory = isHistoryProcessor ? getHistoryProcessor().getHistory() : new INDArray[] {ninput};

            Transition<Integer> trans = new Transition(mdpRunner.getHistory(), action, accuReward, stepReply.isDone(), nhistory[0]);
            getExpReplay().store(trans);

            int historyLength = isHistoryProcessor ? getHistoryProcessor().getConf().getHistoryLength() : 1;
            int updateStart = getConfiguration().getUpdateStart()
                    + ((getConfiguration().getBatchSize() + historyLength) * skipFrame);
            if (getStepCounter() > updateStart) {
                Pair<INDArray, INDArray> targets = setTarget(getExpReplay().getBatch());
                getCurrentDQN().fit(targets.getFirst(), targets.getSecond());
            }

            mdpRunner.setHistory(nhistory);
            accuReward = 0;
        }


        return new QLStepReturn<O>(mdpRunner.getMaxQ(), getCurrentDQN().getLatestScore(), stepReply);

    }


    protected Pair<INDArray, INDArray> setTarget(ArrayList<Transition<Integer>> transitions) {
        if (transitions.size() == 0)
            throw new IllegalArgumentException("too few transitions");

        boolean isHistoryProcessor = getHistoryProcessor() != null;

        int size = transitions.size();

        int[] shape = isHistoryProcessor
                ? getMdp().getObservationSpace().getShape()
                : getHistoryProcessor().getConf().getShape();
        int[] nshape = makeShape(size, shape);
        INDArray obs = Nd4j.create(nshape);
        INDArray nextObs = Nd4j.create(nshape);
        int[] actions = new int[size];
        boolean[] areTerminal = new boolean[size];

        for (int i = 0; i < size; i++) {
            Transition<Integer> trans = transitions.get(i);
            areTerminal[i] = trans.isTerminal();
            actions[i] = trans.getAction();

            INDArray[] obsArray = trans.getObservation();
            if (isHistoryProcessor) {
                // HistoryProcessor requirement: an observation must be 2D grayscale image.
                // The history processor will keep the (IHistoryProcessor.Configuration.historyLength) previous images
                // in the "depth" dimension and feed them to the NN
                for (int j = 0; j < obsArray.length; j++) {
                    obs.put(new INDArrayIndex[] {NDArrayIndex.point(i), NDArrayIndex.point(j)}, obsArray[j]);
                }
            } else {
                obs.putRow(i, obsArray[0]);
            }

            INDArray[] nextObsArray = Transition.append(trans.getObservation(), trans.getNextObservation());
            if (isHistoryProcessor) {
                // HistoryProcessor requirement: an observation must be 2D grayscale image.
                // The history processor will keep the (IHistoryProcessor.Configuration.historyLength) previous images
                // in the "depth" dimension and feed them to the NN
                for (int j = 0; j < nextObsArray.length; j++) {
                    nextObs.put(new INDArrayIndex[] {NDArrayIndex.point(i), NDArrayIndex.point(j)}, nextObsArray[j]);
                }
            } else {
                nextObs.putRow(i, nextObsArray[0]);
            }
        }
        if (getHistoryProcessor() != null) {
            obs.muli(1.0 / getHistoryProcessor().getScale());
            nextObs.muli(1.0 / getHistoryProcessor().getScale());
        }

        INDArray dqnOutputAr = dqnOutput(obs);

        INDArray dqnOutputNext = dqnOutput(nextObs);
        INDArray targetDqnOutputNext = null;

        INDArray tempQ = null;
        INDArray getMaxAction = null;
        if (getConfiguration().isDoubleDQN()) {
            targetDqnOutputNext = targetDqnOutput(nextObs);
            getMaxAction = Nd4j.argMax(dqnOutputNext, 1);
        } else {
            tempQ = Nd4j.max(dqnOutputNext, 1);
        }


        for (int i = 0; i < size; i++) {
            double yTar = transitions.get(i).getReward();
            if (!areTerminal[i]) {
                double q = 0;
                if (getConfiguration().isDoubleDQN()) {
                    q += targetDqnOutputNext.getDouble(i, getMaxAction.getInt(i));
                } else
                    q += tempQ.getDouble(i);

                yTar += getConfiguration().getGamma() * q;

            }



            double previousV = dqnOutputAr.getDouble(i, actions[i]);
            double lowB = previousV - getConfiguration().getErrorClamp();
            double highB = previousV + getConfiguration().getErrorClamp();
            double clamped = Math.min(highB, Math.max(yTar, lowB));

            dqnOutputAr.putScalar(i, actions[i], clamped);
        }

        return new Pair(obs, dqnOutputAr);
    }

}
