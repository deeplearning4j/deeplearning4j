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

package org.deeplearning4j.spark.impl.common.score;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.layers.variational.VariationalAutoencoder;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Function to calculate the scores (reconstruction probability or log probability) for a variational autoencoder.<br>
 * Note that scoring is batched for computational efficiency.<br>
 *
 * @param <K> Type of key, associated with each example. Used to keep track of which score belongs to which example
 * @author Alex Black
 */
public abstract class BaseVaeReconstructionProbWithKeyFunctionAdapter<K> extends BaseVaeScoreWithKeyFunctionAdapter<K> {

    private final boolean useLogProbability;
    private final int numSamples;

    /**
     * @param params                 MultiLayerNetwork parameters
     * @param jsonConfig             MultiLayerConfiguration, as json
     * @param useLogProbability      If true: use log probability. False: use raw probability.
     * @param batchSize              Batch size to use when scoring
     * @param numSamples             Number of samples to use when calling {@link VariationalAutoencoder#reconstructionLogProbability(INDArray, int)}
     */
    public BaseVaeReconstructionProbWithKeyFunctionAdapter(Broadcast<INDArray> params, Broadcast<String> jsonConfig,
                    boolean useLogProbability, int batchSize, int numSamples) {
        super(params, jsonConfig, batchSize);
        this.useLogProbability = useLogProbability;
        this.numSamples = numSamples;
    }

    @Override
    public INDArray computeScore(VariationalAutoencoder vae, INDArray toScore) {
        if (useLogProbability) {
            return vae.reconstructionLogProbability(toScore, numSamples);
        } else {
            return vae.reconstructionProbability(toScore, numSamples);
        }
    }
}
