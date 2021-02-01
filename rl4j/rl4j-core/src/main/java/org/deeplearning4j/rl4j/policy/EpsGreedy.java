/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.policy;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.environment.IActionSchema;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/24/16.
 *
 * An epsilon greedy policy choose the next action
 * - randomly with epsilon probability
 * - deleguate it to constructor argument 'policy' with (1-epsilon) probability.
 *
 * epislon is annealed to minEpsilon over epsilonNbStep steps
 *
 */
@Slf4j
public class EpsGreedy<A> extends Policy<A> {

    final private INeuralNetPolicy<A> policy;
    final private int annealingStart;
    final private int epsilonNbStep;
    final private Random rnd;
    final private double minEpsilon;

    private final IActionSchema<A> actionSchema;

    final private MDP<Encodable, A, ActionSpace<A>> mdp;
    final private IEpochTrainer learning;

    // Using agent's (learning's) step count is incorrect; frame skipping makes epsilon's value decrease too quickly
    private int annealingStep = 0;

    @Deprecated
    public <OBSERVATION extends Encodable, AS extends ActionSpace<A>> EpsGreedy(Policy<A> policy,
                                                                                MDP<Encodable, A, ActionSpace<A>> mdp,
                                                                                int annealingStart,
                                                                                int epsilonNbStep,
                                                                                Random rnd,
                                                                                double minEpsilon,
                                                                                IEpochTrainer learning) {
        this.policy = policy;
        this.mdp = mdp;
        this.annealingStart = annealingStart;
        this.epsilonNbStep = epsilonNbStep;
        this.rnd = rnd;
        this.minEpsilon = minEpsilon;
        this.learning = learning;

        this.actionSchema = null;
    }

    public EpsGreedy(@NonNull Policy<A> policy, @NonNull IActionSchema<A> actionSchema, double minEpsilon, int annealingStart, int epsilonNbStep) {
        this(policy, actionSchema, minEpsilon, annealingStart, epsilonNbStep, null);
    }

    @Builder
    public EpsGreedy(@NonNull INeuralNetPolicy<A> policy, @NonNull IActionSchema<A> actionSchema, double minEpsilon, int annealingStart, int epsilonNbStep, Random rnd) {
        this.policy = policy;

        this.rnd = rnd == null ? Nd4j.getRandom() : rnd;
        this.minEpsilon = minEpsilon;
        this.annealingStart = annealingStart;
        this.epsilonNbStep = epsilonNbStep;
        this.actionSchema = actionSchema;

        this.mdp = null;
        this.learning = null;
    }

    public EpsGreedy(INeuralNetPolicy<A> policy, IActionSchema<A> actionSchema, @NonNull Configuration configuration, Random rnd) {
        this(policy, actionSchema, configuration.getMinEpsilon(), configuration.getAnnealingStart(), configuration.getEpsilonNbStep(), rnd);
    }

    public IOutputNeuralNet getNeuralNet() {
        return policy.getNeuralNet();
    }

    @Deprecated
    public A nextAction(INDArray input) {

        double ep = getEpsilon();
        if(actionSchema != null) {
            // Only legacy classes should pass here.
            throw new RuntimeException("nextAction(Observation observation) should be called when using a AgentLearner");
        }

        if (learning.getStepCount() % 500 == 1)
            log.info("EP: " + ep + " " + learning.getStepCount());
        if (rnd.nextDouble() > ep)
            return policy.nextAction(input);
        else
            return mdp.getActionSpace().randomAction();
    }

    public A nextAction(Observation observation) {
        // FIXME: remove if() and content once deprecated methods are removed.
        if(actionSchema == null) {
            return this.nextAction(observation.getChannelData(0));
        }

        double ep = getEpsilon();
        if (annealingStep % 500 == 1) {
            log.info("EP: " + ep + " " + annealingStep);
        }

        ++annealingStep;

        // TODO: This is a temporary solution while something better is developed
        if (rnd.nextDouble() > ep) {
            return policy.nextAction(observation);
        }
        // With RNNs the neural net must see *all* observations
        if(getNeuralNet().isRecurrent()) {
            policy.nextAction(observation); // Make the RNN see the observation
        }
        return actionSchema.getRandomAction();
    }

    public double getEpsilon() {
        int step = actionSchema != null ? annealingStep : learning.getStepCount();
        return Math.min(1.0, Math.max(minEpsilon, 1.0 - (step - annealingStart) * 1.0 / epsilonNbStep));
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        @Builder.Default
        final int annealingStart = 0;

        final int epsilonNbStep;
        final double minEpsilon;
    }
}
