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
            INDArray weights = weightParamView.reshape('f', nIn, nOut);
            if (weightInit instanceof WeightInitEmbedding) {
                // WeightInitEmbedding copies pre-trained weights directly — it must
                // receive the full [vocabSize, vectorSize] shape, not row-by-row.
                weightInit.init(nIn, nOut, shape, IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, weights);
            } else {
                // Initialize row-by-row so each call operates on nOut elements.
                // For typical nOut values (<= 1024) this stays single-threaded in the
                // native CPU random kernel, avoiding a data race on the shared coords
                // buffer in the single-arg execTransform path that is used by
                // UniformDistribution (XAVIER_UNIFORM).  Initializing the whole
                // nIn*nOut matrix at once with nIn large enough (>= 1024/nOut rows)
                // causes the kernel to spawn multiple threads that all write to the
                // same stack-allocated coords array, producing non-deterministic results.
                long[] rowShape = new long[]{1, nOut};
                for (long i = 0; i < nIn; i++) {
                    INDArray row = weights.getRow(i);
                    weightInit.init(1, nOut, rowShape, IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, row);
                }
            }
            return weights;
        } else {
            return WeightInitUtil.reshapeWeights(shape, weightParamView);
        }
    }

}
