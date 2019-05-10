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

package org.deeplearning4j.rl4j.mdp;


import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 * An interface that ensure an environment is expressible as a
 * Markov Decsision Process. This implementation follow the gym model.
 * It works based on side effects which is perfect for imperative simulation.
 *
 * A bit sad that it doesn't use efficiently stateful mdp that could be rolled back
 * in a "functionnal manner" if step return a mdp
 *
 */
public interface MDP<O, A, AS extends ActionSpace<A>> {

    ObservationSpace<O> getObservationSpace();

    AS getActionSpace();

    O reset();

    void close();

    StepReply<O> step(A action);

    boolean isDone();

    MDP<O, A, AS> newInstance();

}
