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

package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/19/16.
 *
 * A common interface that any training method should implement
 */
public interface ILearning<O extends Encodable, A, AS extends ActionSpace<A>> extends StepCountable {

    Policy<O, A> getPolicy();

    void train();

    int getStepCounter();

    LConfiguration getConfiguration();

    MDP<O, A, AS> getMdp();


    interface LConfiguration {

        int getSeed();

        int getMaxEpochStep();

        int getMaxStep();

        double getGamma();
    }

}
