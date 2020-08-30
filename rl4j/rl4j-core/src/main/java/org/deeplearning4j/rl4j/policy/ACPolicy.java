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

import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.ac.ActorCriticCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticSeparate;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * A stochastic policy thats explore the environment based on
 * the softmax output of the actor critic, but objects constructed
 * with a {@link Random} argument of null return the max only.
 */
public class ACPolicy<OBSERVATION extends Encodable> extends Policy<Integer> {

    final private IActorCritic actorCritic;
    Random rnd;

    public ACPolicy(IActorCritic actorCritic) {
        this(actorCritic, Nd4j.getRandom());
    }
    public ACPolicy(IActorCritic actorCritic, Random rnd) {
        this.actorCritic = actorCritic;
        this.rnd = rnd;
    }

    public static <OBSERVATION extends Encodable> ACPolicy<OBSERVATION> load(String path) throws IOException {
        return new ACPolicy<>(ActorCriticCompGraph.load(path));
    }
    public static <OBSERVATION extends Encodable> ACPolicy<OBSERVATION> load(String path, Random rnd) throws IOException {
        return new ACPolicy<>(ActorCriticCompGraph.load(path), rnd);
    }

    public static <OBSERVATION extends Encodable> ACPolicy<OBSERVATION> load(String pathValue, String pathPolicy) throws IOException {
        return new ACPolicy<>(ActorCriticSeparate.load(pathValue, pathPolicy));
    }
    public static <OBSERVATION extends Encodable> ACPolicy<OBSERVATION> load(String pathValue, String pathPolicy, Random rnd) throws IOException {
        return new ACPolicy<>(ActorCriticSeparate.load(pathValue, pathPolicy), rnd);
    }

    public IActorCritic getNeuralNet() {
        return actorCritic;
    }

    @Override
    public Integer nextAction(Observation obs) {
        return nextAction(obs.getData());
    }

    public Integer nextAction(INDArray input) {
        INDArray output = actorCritic.outputAll(input)[1];
        if (rnd == null) {
            return Learning.getMaxAction(output);
        }
        float rVal = rnd.nextFloat();
        for (int i = 0; i < output.length(); i++) {
            //System.out.println(i + " " + rVal + " " + output.getFloat(i));
            if (rVal < output.getFloat(i)) {
                return i;
            } else
                rVal -= output.getFloat(i);
        }

        throw new RuntimeException("Output from network is not a probability distribution: " + output);
    }

    public void save(String filename) throws IOException {
        actorCritic.save(filename);
    }

    public void save(String filenameValue, String filenamePolicy) throws IOException {
        actorCritic.save(filenameValue, filenamePolicy);
    }

}
