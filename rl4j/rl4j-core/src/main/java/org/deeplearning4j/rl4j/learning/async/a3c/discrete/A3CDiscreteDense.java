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

package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.*;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/8/16.
 *
 * Training for A3C in the Discrete Domain
 *
 * We use specifically the Separate version because
 * the model is too small to have enough benefit by sharing layers
 *
 */
public class A3CDiscreteDense<O extends Encodable> extends A3CDiscrete<O> {

    public A3CDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, IActorCritic IActorCritic, A3CConfiguration conf,
                    DataManager dataManager) {
        super(mdp, IActorCritic, conf, dataManager);
    }

    public A3CDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, ActorCriticFactorySeparate factory,
                    A3CConfiguration conf, DataManager dataManager) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                        dataManager);
    }

    public A3CDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp,
                    ActorCriticFactorySeparateStdDense.Configuration netConf, A3CConfiguration conf,
                    DataManager dataManager) {
        this(mdp, new ActorCriticFactorySeparateStdDense(netConf), conf, dataManager);
    }

    public A3CDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory,
                    A3CConfiguration conf, DataManager dataManager) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                        dataManager);
    }

    public A3CDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp,
                    ActorCriticFactoryCompGraphStdDense.Configuration netConf, A3CConfiguration conf,
                    DataManager dataManager) {
        this(mdp, new ActorCriticFactoryCompGraphStdDense(netConf), conf, dataManager);
    }

}
