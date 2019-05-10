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

package org.deeplearning4j.rl4j.learning.sync.qlearning;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.sync.ExpReplay;
import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.SyncLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager.StatEntry;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/19/16.
 * <p>
 * Mother class for QLearning in the Discrete domain and
 * hopefully one day for the  Continuous domain.
 */
@Slf4j
public abstract class QLearning<O extends Encodable, A, AS extends ActionSpace<A>>
                extends SyncLearning<O, A, AS, IDQN> {

    @Getter
    final private IExpReplay<A> expReplay;

    public QLearning(QLConfiguration conf) {
        super(conf);
        expReplay = new ExpReplay<>(conf.getExpRepMaxSize(), conf.getBatchSize(), conf.getSeed());
    }

    protected abstract EpsGreedy<O, A, AS> getEgPolicy();

    public abstract MDP<O, A, AS> getMdp();

    protected abstract IDQN getCurrentDQN();

    protected abstract IDQN getTargetDQN();

    protected abstract void setTargetDQN(IDQN dqn);

    protected INDArray dqnOutput(INDArray input) {
        return getCurrentDQN().output(input);
    }

    protected INDArray targetDqnOutput(INDArray input) {
        return getTargetDQN().output(input);
    }

    protected void updateTargetNetwork() {
        log.info("Update target network");
        setTargetDQN(getCurrentDQN().clone());
    }


    public IDQN getNeuralNet() {
        return getCurrentDQN();
    }

    public abstract QLConfiguration getConfiguration();

    protected abstract void preEpoch();

    protected abstract void postEpoch();

    protected abstract QLStepReturn<O> trainStep(O obs);

    protected StatEntry trainEpoch() {
        InitMdp<O> initMdp = initMdp();
        O obs = initMdp.getLastObs();

        double reward = initMdp.getReward();
        int step = initMdp.getSteps();

        Double startQ = Double.NaN;
        double meanQ = 0;
        int numQ = 0;
        List<Double> scores = new ArrayList<>();
        while (step < getConfiguration().getMaxEpochStep() && !getMdp().isDone()) {

            if (getStepCounter() % getConfiguration().getTargetDqnUpdateFreq() == 0) {
                updateTargetNetwork();
            }

            QLStepReturn<O> stepR = trainStep(obs);

            if (!stepR.getMaxQ().isNaN()) {
                if (startQ.isNaN())
                    startQ = stepR.getMaxQ();
                numQ++;
                meanQ += stepR.getMaxQ();
            }

            if (stepR.getScore() != 0)
                scores.add(stepR.getScore());

            reward += stepR.getStepReply().getReward();
            obs = stepR.getStepReply().getObservation();
            incrementStep();
            step++;
        }

        meanQ /= (numQ + 0.001); //avoid div zero


        StatEntry statEntry = new QLStatEntry(getStepCounter(), getEpochCounter(), reward, step, scores,
                        getEgPolicy().getEpsilon(), startQ, meanQ);

        return statEntry;

    }

    @AllArgsConstructor
    @Builder
    @Value
    public static class QLStatEntry implements StatEntry {
        int stepCounter;
        int epochCounter;
        double reward;
        int episodeLength;
        List<Double> scores;
        float epsilon;
        double startQ;
        double meanQ;
    }

    @AllArgsConstructor
    @Builder
    @Value
    public static class QLStepReturn<O> {
        Double maxQ;
        double score;
        StepReply<O> stepReply;

    }

    @Data
    @AllArgsConstructor
    @Builder
    @EqualsAndHashCode(callSuper = false)
    @JsonDeserialize(builder = QLConfiguration.QLConfigurationBuilder.class)
    public static class QLConfiguration implements LConfiguration {

        int seed;
        int maxEpochStep;
        int maxStep;
        int expRepMaxSize;
        int batchSize;
        int targetDqnUpdateFreq;
        int updateStart;
        double rewardFactor;
        double gamma;
        double errorClamp;
        float minEpsilon;
        int epsilonNbStep;
        boolean doubleDQN;

        @JsonPOJOBuilder(withPrefix = "")
        public static final class QLConfigurationBuilder {
        }
    }


}
