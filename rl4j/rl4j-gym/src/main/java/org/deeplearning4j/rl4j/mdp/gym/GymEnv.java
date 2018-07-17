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

package org.deeplearning4j.rl4j.mdp.gym;


import org.deeplearning4j.gym.Client;
import org.deeplearning4j.gym.ClientFactory;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.HighLowDiscrete;
import org.deeplearning4j.rl4j.space.ObservationSpace;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 *
 * Wrapper over the client of gym-java-client
 *
 */
public class GymEnv<O, A, AS extends ActionSpace<A>> implements MDP<O, A, AS> {

    final public static String GYM_MONITOR_DIR = "/tmp/gym-dqn";

    final private Client<O, A, AS> client;
    final private String envId;
    final private boolean render;
    final private boolean monitor;
    private ActionTransformer actionTransformer = null;
    private boolean done = false;

    public GymEnv(String envId, boolean render, boolean monitor) {
        this.client = ClientFactory.build(envId, render);
        this.envId = envId;
        this.render = render;
        this.monitor = monitor;
        if (monitor)
            client.monitorStart(GYM_MONITOR_DIR, true, false);
    }

    public GymEnv(String envId, boolean render, boolean monitor, int[] actions) {
        this(envId, render, monitor);
        actionTransformer = new ActionTransformer((HighLowDiscrete) getActionSpace(), actions);
    }


    public ObservationSpace<O> getObservationSpace() {
        return client.getObservationSpace();
    }

    public AS getActionSpace() {
        if (actionTransformer == null)
            return (AS) client.getActionSpace();
        else
            return (AS) actionTransformer;
    }

    public StepReply<O> step(A action) {
        StepReply<O> stepRep = client.step(action);
        done = stepRep.isDone();
        return stepRep;
    }

    public boolean isDone() {
        return done;
    }

    public O reset() {
        done = false;
        return client.reset();
    }


    public void upload(String apiKey) {
        client.upload(GYM_MONITOR_DIR, apiKey);
    }

    public void close() {
        if (monitor)
            client.monitorClose();
    }

    public GymEnv<O, A, AS> newInstance() {
        return new GymEnv<O, A, AS>(envId, render, monitor);
    }
}
