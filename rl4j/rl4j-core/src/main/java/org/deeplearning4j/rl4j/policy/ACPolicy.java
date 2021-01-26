/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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
import lombok.NonNull;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.deeplearning4j.rl4j.network.ac.ActorCriticCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticSeparate;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

/**
 * A stochastic policy that, when training, explore the environment based on
 * the softmax output of the actor critic, but objects constructed.
 * Revert to a greedy policy when not training.
 *
 */
public class ACPolicy<OBSERVATION extends Encodable> extends Policy<Integer> {

    final private IOutputNeuralNet neuralNet;
    private final boolean isTraining;
    private final Random rnd;

    @Builder
    public ACPolicy(@NonNull IOutputNeuralNet neuralNet, boolean isTraining, Random rnd) {
        this.neuralNet = neuralNet;
        this.isTraining = isTraining;
        this.rnd = !isTraining || rnd != null ? rnd : Nd4j.getRandom();
    }

    public static <OBSERVATION extends Encodable> ACPolicy<OBSERVATION> load(String path) throws IOException {
        // TODO: Add better load/save support
        return new ACPolicy<>(ActorCriticCompGraph.load(path), false, null);
    }
    public static <OBSERVATION extends Encodable> ACPolicy<OBSERVATION> load(String path, Random rnd) throws IOException {
        // TODO: Add better load/save support
        return new ACPolicy<>(ActorCriticCompGraph.load(path), true, rnd);
    }

    public static <OBSERVATION extends Encodable> ACPolicy<OBSERVATION> load(String pathValue, String pathPolicy) throws IOException {
        return new ACPolicy<>(ActorCriticSeparate.load(pathValue, pathPolicy), false, null);
    }
    public static <OBSERVATION extends Encodable> ACPolicy<OBSERVATION> load(String pathValue, String pathPolicy, Random rnd) throws IOException {
        return new ACPolicy<>(ActorCriticSeparate.load(pathValue, pathPolicy), true, rnd);
    }

    @Deprecated
    public IOutputNeuralNet getNeuralNet() {
        return neuralNet;
    }

    @Override
    public Integer nextAction(Observation obs) {
        // Review: Should ActorCriticPolicy be a training policy only? Why not use the existing greedy policy when not training instead of duplicating the code?
        INDArray output = neuralNet.output(obs).get(CommonOutputNames.ActorCritic.Policy);
        if (!isTraining) {
            return Learning.getMaxAction(output);
        }

        float rVal = rnd.nextFloat();
        for (int i = 0; i < output.length(); i++) {
            if (rVal < output.getFloat(i)) {
                return i;
            } else
                rVal -= output.getFloat(i);
        }

        throw new RuntimeException("Output from network is not a probability distribution: " + output);
    }

    @Deprecated
    public Integer nextAction(INDArray input) {
        INDArray output = ((IActorCritic) neuralNet).outputAll(input)[1];
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
        // TODO: Add better load/save support
        ((IActorCritic) neuralNet).save(filename);
    }

    public void save(String filenameValue, String filenamePolicy) throws IOException {
        // TODO: Add better load/save support
        ((IActorCritic) neuralNet).save(filenameValue, filenamePolicy);
    }

}
