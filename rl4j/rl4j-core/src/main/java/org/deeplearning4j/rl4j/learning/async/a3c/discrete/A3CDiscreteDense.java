/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
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

import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.*;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/8/16.
 * <p>
 * Training for A3C in the Discrete Domain
 * <p>
 * We use specifically the Separate version because
 * the model is too small to have enough benefit by sharing layers
 */
public class A3CDiscreteDense<OBSERVATION extends Encodable> extends A3CDiscrete<OBSERVATION> {

    @Deprecated
    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IActorCritic IActorCritic, A3CConfiguration conf,
                            IDataManager dataManager) {
        this(mdp, IActorCritic, conf);
        addListener(new DataManagerTrainingListener(dataManager));
    }

    @Deprecated
    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IActorCritic actorCritic, A3CConfiguration conf) {
        super(mdp, actorCritic, conf.toLearningConfiguration());
    }

    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IActorCritic actorCritic, A3CLearningConfiguration conf) {
        super(mdp, actorCritic, conf);
    }

    @Deprecated
    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactorySeparate factory,
                            A3CConfiguration conf, IDataManager dataManager) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                dataManager);
    }

    @Deprecated
    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactorySeparate factory,
                            A3CConfiguration conf) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactorySeparate factory,
                            A3CLearningConfiguration conf) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    @Deprecated
    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp,
                            ActorCriticFactorySeparateStdDense.Configuration netConf, A3CConfiguration conf,
                            IDataManager dataManager) {
        this(mdp, new ActorCriticFactorySeparateStdDense(netConf.toNetworkConfiguration()), conf, dataManager);
    }

    @Deprecated
    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp,
                            ActorCriticFactorySeparateStdDense.Configuration netConf, A3CConfiguration conf) {
        this(mdp, new ActorCriticFactorySeparateStdDense(netConf.toNetworkConfiguration()), conf);
    }

    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp,
                            ActorCriticDenseNetworkConfiguration netConf, A3CLearningConfiguration conf) {
        this(mdp, new ActorCriticFactorySeparateStdDense(netConf), conf);
    }

    @Deprecated
    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory,
                            A3CConfiguration conf, IDataManager dataManager) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                dataManager);
    }

    @Deprecated
    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory,
                            A3CConfiguration conf) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    public A3CDiscreteDense(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory,
                            A3CLearningConfiguration conf) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }
}
