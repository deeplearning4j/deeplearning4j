/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.TDTargetAlgorithm;

import lombok.Setter;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QNetworkSource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

/**
 * The base of all TD target calculation algorithms that use deep learning.
 *
 * @author Alexandre Boulanger
 */
public abstract class BaseTDTargetAlgorithm implements ITDTargetAlgorithm<Integer> {

    protected final QNetworkSource qNetworkSource;
    protected final double gamma;

    private final double errorClamp;
    private final boolean isClamped;

    @Setter
    private int[] nShape; // TODO: Remove once we use DataSets in observations

    /**
     *
     * @param qNetworkSource The source of the Q-Network
     * @param gamma The discount factor
     * @param errorClamp Will prevent the new Q-Value from being farther than <i>errorClamp</i> away from the previous value. Double.NaN will disable the clamping.
     */
    protected BaseTDTargetAlgorithm(QNetworkSource qNetworkSource, double gamma, double errorClamp) {
        this.qNetworkSource = qNetworkSource;
        this.gamma = gamma;

        this.errorClamp = errorClamp;
        isClamped = !Double.isNaN(errorClamp);
    }

    /**
     *
     * @param qNetworkSource The source of the Q-Network
     * @param gamma The discount factor
     * Note: Error clamping is disabled with this ctor
     */
    protected BaseTDTargetAlgorithm(QNetworkSource qNetworkSource, double gamma) {
        this(qNetworkSource, gamma, Double.NaN);
    }

    /**
     * Called just before the calculation starts
     * @param observations A INDArray of all observations stacked on dimension 0
     * @param nextObservations A INDArray of all next observations stacked on dimension 0
     */
    protected void initComputation(INDArray observations, INDArray nextObservations) {
        // Do nothing
    }

    /**
     * Compute the new estimated Q-Value for every transition in the batch
     *
     * @param batchIdx The index in the batch of the current transition
     * @param reward The reward of the current transition
     * @param isTerminal True if it's the last transition of the "game"
     * @return The estimated Q-Value
     */
    protected abstract double computeTarget(int batchIdx, double reward, boolean isTerminal);

    @Override
    public DataSet computeTDTargets(List<Transition<Integer>> transitions) {

        int size = transitions.size();

        INDArray observations = Nd4j.create(nShape);
        INDArray nextObservations = Nd4j.create(nShape);

        // TODO: Remove once we use DataSets in observations
        for (int i = 0; i < size; i++) {
            Transition<Integer> trans = transitions.get(i);

            INDArray[] obsArray = trans.getObservation();
            if (observations.rank() == 2) {
                observations.putRow(i, obsArray[0]);
            } else {
                for (int j = 0; j < obsArray.length; j++) {
                    observations.put(new INDArrayIndex[] {NDArrayIndex.point(i), NDArrayIndex.point(j)}, obsArray[j]);
                }
            }

            INDArray[] nextObsArray = Transition.append(trans.getObservation(), trans.getNextObservation());
            if (nextObservations.rank() == 2) {
                nextObservations.putRow(i, nextObsArray[0]);
            } else {
                for (int j = 0; j < nextObsArray.length; j++) {
                    nextObservations.put(new INDArrayIndex[] {NDArrayIndex.point(i), NDArrayIndex.point(j)}, nextObsArray[j]);
                }
            }
        }

        initComputation(observations, nextObservations);

        INDArray updatedQValues = qNetworkSource.getQNetwork().output(observations);

        for (int i = 0; i < size; ++i) {
            Transition<Integer> transition = transitions.get(i);
            double yTarget = computeTarget(i, transition.getReward(), transition.isTerminal());

            if(isClamped) {
                double previousQValue = updatedQValues.getDouble(i, transition.getAction());
                double lowBound = previousQValue - errorClamp;
                double highBound = previousQValue + errorClamp;
                yTarget = Math.min(highBound, Math.max(yTarget, lowBound));
            }
            updatedQValues.putScalar(i, transition.getAction(), yTarget);
        }

        return new org.nd4j.linalg.dataset.DataSet(observations, updatedQValues);
    }
}
