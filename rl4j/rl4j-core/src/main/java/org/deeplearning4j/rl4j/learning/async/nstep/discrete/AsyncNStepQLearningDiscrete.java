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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.configuration.AsyncQLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public abstract class AsyncNStepQLearningDiscrete<O extends Encodable>
        extends AsyncLearning<O, Integer, DiscreteSpace, IDQN> {

    @Getter
    final public AsyncQLearningConfiguration configuration;
    @Getter
    final private MDP<O, Integer, DiscreteSpace> mdp;
    @Getter
    final private AsyncGlobal<IDQN> asyncGlobal;


    public AsyncNStepQLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, AsyncQLearningConfiguration conf) {
        this.mdp = mdp;
        this.configuration = conf;
        this.asyncGlobal = new AsyncGlobal<>(dqn, conf);
    }

    @Override
    public AsyncThread newThread(int i, int deviceNum) {
        return new AsyncNStepQLearningThreadDiscrete(mdp.newInstance(), asyncGlobal, configuration, getListeners(), i, deviceNum);
    }

    public IDQN getNeuralNet() {
        return asyncGlobal.getTarget();
    }

    public IPolicy<O, Integer> getPolicy() {
        return new DQNPolicy<O>(getNeuralNet());
    }

    @Data
    @AllArgsConstructor
    @Builder
    @EqualsAndHashCode(callSuper = false)
    public static class AsyncNStepQLConfiguration {

        Integer seed;
        int maxEpochStep;
        int maxStep;
        int numThread;
        int nstep;
        int targetDqnUpdateFreq;
        int updateStart;
        double rewardFactor;
        double gamma;
        double errorClamp;
        float minEpsilon;
        int epsilonNbStep;

        public AsyncQLearningConfiguration toLearningConfiguration() {
            return AsyncQLearningConfiguration.builder()
                    .seed(new Long(seed))
                    .maxEpochStep(maxEpochStep)
                    .maxStep(maxStep)
                    .numThreads(numThread)
                    .nStep(nstep)
                    .targetDqnUpdateFreq(targetDqnUpdateFreq)
                    .updateStart(updateStart)
                    .rewardFactor(rewardFactor)
                    .gamma(gamma)
                    .errorClamp(errorClamp)
                    .minEpsilon(minEpsilon)
                    .build();
        }

    }

}
