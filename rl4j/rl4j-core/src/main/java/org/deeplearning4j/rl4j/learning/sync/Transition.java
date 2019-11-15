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

package org.deeplearning4j.rl4j.learning.sync;

import lombok.Value;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 *
 * A transition is a SARS tuple
 * State, Action, Reward, (isTerminal), State
 */
@Value
public class Transition<A> {

    Observation observation;
    A action;
    double reward;
    boolean isTerminal;
    INDArray nextObservation;

    public Transition(Observation observation, A action, double reward, boolean isTerminal, Observation nextObservation) {
        this.observation = observation;
        this.action = action;
        this.reward = reward;
        this.isTerminal = isTerminal;

        // For the next observation, only keep the latest frame to save memory. The full nextObservation will be re-build
        // from observation when needed.
        long[] nextObservationShape = nextObservation.getData().shape().clone();
        nextObservationShape[0] = 1;
        this.nextObservation = nextObservation.getData().get(new INDArrayIndex[] {NDArrayIndex.point(0)}).reshape(nextObservationShape);
    }

    private Transition(Observation observation, A action, double reward, boolean isTerminal, INDArray nextObservation) {
        this.observation = observation;
        this.action = action;
        this.reward = reward;
        this.isTerminal = isTerminal;
        this.nextObservation = nextObservation;
    }

    /**
     * concat an array history into a single INDArry of as many channel
     * as element in the history array
     * @param history the history to concat
     * @return the multi-channel INDArray
     */
    public static INDArray concat(INDArray[] history) {
        INDArray arr = Nd4j.concat(0, history);
        return arr;
    }

    /**
     * Duplicate this transition
     * @return this transition duplicated
     */
    public Transition<A> dup() {
        Observation dupObservation = observation.dup();
        INDArray nextObs = nextObservation.dup();

        return new Transition<A>(dupObservation, action, reward, isTerminal, nextObs);
    }

    public static <A> INDArray buildStackedObservations(List<Transition<A>> transitions) {
        int size = transitions.size();
        long[] shape = getShape(transitions);

        INDArray[] array = new INDArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = transitions.get(i).getObservation().getData();
        }

        return  Nd4j.concat(0, array).reshape(shape);
    }

    public static <A> INDArray buildStackedNextObservations(List<Transition<A>> transitions) {
        int size = transitions.size();
        long[] shape = getShape(transitions);

        INDArray[] array = new INDArray[size];

        for (int i = 0; i < size; i++) {
            Transition<A> trans = transitions.get(i);
            INDArray obs = trans.getObservation().getData();
            long historyLength = obs.shape()[0];

            if(historyLength != 1) {
                INDArray historyPart = obs.get(new INDArrayIndex[]{NDArrayIndex.interval(0, historyLength - 1)});
                array[i] = Nd4j.concat(0, trans.getNextObservation(), historyPart);
            }
            else {
                array[i] = trans.getNextObservation();
            }
        }

        return  Nd4j.concat(0, array).reshape(shape);
    }

    private static <A> long[] getShape(List<Transition<A>> transitions) {
        INDArray observations = transitions.get(0).getObservation().getData();
        long[] observationShape = observations.shape();
        long[] stackedShape;
        if(observationShape[0] == 1) {
            // FIXME: Currently RL4J doesn't support 1D observations. So if we have a shape with 1 in the first dimension, we can use that dimension and don't need to add another one.
            stackedShape = new long[observationShape.length];
            System.arraycopy(observationShape, 0, stackedShape, 0, observationShape.length);
        }
        else {
            stackedShape = new long[observationShape.length + 1];
            System.arraycopy(observationShape, 1, stackedShape, 2, observationShape.length - 1);
            stackedShape[1] = observationShape[1];
        }
        stackedShape[0] = transitions.size();

        return stackedShape;
    }

}
