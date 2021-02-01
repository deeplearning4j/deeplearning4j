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

package org.deeplearning4j.spark.impl.multilayer.scoring;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.common.score.BaseVaeScoreWithKeyFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple2;

import java.util.Iterator;


/**
 * Function to calculate the reconstruction error for a variational autoencoder, that is the first layer in a
 * MultiLayerNetwork.<br>
 * Note that the VAE must be using a loss function, not a {@link org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution}<br>
 * Also note that scoring is batched for computational efficiency.<br>
 *
 * @author Alex Black
 * @see VaeReconstructionProbWithKeyFunction
 */
public class VaeReconstructionErrorWithKeyFunction<K> extends BaseVaeScoreWithKeyFunction<K> {

    /**
     * @param params            MultiLayerNetwork parameters
     * @param jsonConfig        MultiLayerConfiguration, as json
     * @param batchSize         Batch size to use when scoring
     */
    public VaeReconstructionErrorWithKeyFunction(Broadcast<INDArray> params, Broadcast<String> jsonConfig,
                    int batchSize) {
        super(params, jsonConfig, batchSize);
    }

    @Override
    public VariationalAutoencoder getVaeLayer() {
        MultiLayerNetwork network =
                        new MultiLayerNetwork(MultiLayerConfiguration.fromJson((String) jsonConfig.getValue()));
        network.init();
        INDArray val = ((INDArray) params.value()).unsafeDuplication();
        if (val.length() != network.numParams(false))
            throw new IllegalStateException(
                            "Network did not have same number of parameters as the broadcast set parameters");
        network.setParameters(val);

        Layer l = network.getLayer(0);
        if (!(l instanceof VariationalAutoencoder)) {
            throw new RuntimeException(
                            "Cannot use VaeReconstructionErrorWithKeyFunction on network that doesn't have a VAE "
                                            + "layer as layer 0. Layer type: " + l.getClass());
        }
        return (VariationalAutoencoder) l;
    }

    @Override
    public INDArray computeScore(VariationalAutoencoder vae, INDArray toScore) {
        return vae.reconstructionError(toScore);
    }
}
