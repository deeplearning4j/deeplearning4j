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

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

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
@AllArgsConstructor
@Slf4j
public class EpsGreedy<O extends Encodable, A, AS extends ActionSpace<A>> extends Policy<O, A> {

    final private Policy<O, A> policy;
    final private MDP<O, A, AS> mdp;
    final private int updateStart;
    final private int epsilonNbStep;
    final private Random rd;
    final private float minEpsilon;
    final private StepCountable learning;

    public NeuralNet getNeuralNet() {
        return policy.getNeuralNet();
    }

    public A nextAction(INDArray input) {

        float ep = getEpsilon();
        if (learning.getStepCounter() % 500 == 1)
            log.info("EP: " + ep + " " + learning.getStepCounter());
        if (rd.nextFloat() > ep)
            return policy.nextAction(input);
        else
            return mdp.getActionSpace().randomAction();


    }

    public float getEpsilon() {
        return Math.min(1f, Math.max(minEpsilon, 1f - (learning.getStepCounter() - updateStart) * 1f / epsilonNbStep));
    }
}
