/* ******************************************************************************
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
package org.deeplearning4j.nn.params;

import lombok.val;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Parameter initializer for EmbeddingLayer and EmbeddingSequenceLayer
 *
 * @author Alex Black
 */
public class EmbeddingLayerParamInitializer extends DefaultParamInitializer {

    private static final EmbeddingLayerParamInitializer INSTANCE = new EmbeddingLayerParamInitializer();

    public static EmbeddingLayerParamInitializer getInstance() {
        return INSTANCE;
    }



    protected INDArray createWeightMatrix(long nIn, long nOut, IWeightInit weightInit,
                                          INDArray weightParamView, boolean initializeParameters) {
        val shape = new long[] {nIn, nOut};

        if (initializeParameters) {
            INDArray ret = weightInit.init(1, //Fan in - note that fanIn=1 for embedding layer... if we used layer nIn (i.e., vocab size) the init would depend on vocab size (which doesn't make sense)
                    nOut, //Fan out
                    shape, IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, weightParamView);
            return ret;
        } else {
            return WeightInitUtil.reshapeWeights(shape, weightParamView);
        }
    }

}
