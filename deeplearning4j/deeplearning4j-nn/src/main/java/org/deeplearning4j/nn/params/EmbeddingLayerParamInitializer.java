/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.nn.params;

import lombok.val;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.nn.weights.embeddings.WeightInitEmbedding;
import org.nd4j.linalg.api.ndarray.INDArray;

public class EmbeddingLayerParamInitializer extends DefaultParamInitializer {

    private static final EmbeddingLayerParamInitializer INSTANCE = new EmbeddingLayerParamInitializer();

    public static EmbeddingLayerParamInitializer getInstance() {
        return INSTANCE;
    }



    protected INDArray createWeightMatrix(long nIn, long nOut, IWeightInit weightInit,
                                          INDArray weightParamView, boolean initializeParameters) {
        val shape = new long[] {nIn, nOut};

        if (initializeParameters) {
            if (weightInit instanceof WeightInitEmbedding) {
                // WeightInitEmbedding copies pre-trained weights directly — it must
                // receive the full [vocabSize, vectorSize] shape, not row-by-row.
                INDArray weights = weightParamView.reshape(IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, nIn, nOut);
                weightInit.init(nIn, nOut, shape, IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, weights);
                return weights;
            } else {
                // Initialize using standard fan-in=1 approach for embedding layers.
                // fanIn=1 is correct here: vocab size shouldn't affect the init scale,
                // since each row is a separate lookup that doesn't accumulate over vocab.
                INDArray ret = weightInit.init(1, //Fan in
                        nOut, //Fan out
                        shape, IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, weightParamView);
                return ret;
            }
        } else {
            return WeightInitUtil.reshapeWeights(shape, weightParamView);
        }
    }

}
