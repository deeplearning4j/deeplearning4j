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
package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

@Data
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class DotProductAttentionLayer extends SameDiffLayer {
    private double scaleFactor;
    private double dropoutProbability;
    private int scoreMode;
    private boolean useCausalMask;
    private boolean withWeights;
    private boolean training;
    private long nIn;
    private long nOut;
    public final static String Q_KEY = "q";
    public final static String K_KEY = "k";
    public final static String V_KEY = "v";

    public final static String Q_MASK_KEY = "q_mask";
    public final static String K_MASK_KEY = "k_mask";
    public final static String V_MASK_KEY = "v_mask";

    public DotProductAttentionLayer(Builder builder) {
        super(builder);
        nIn = builder.nIn;
        nOut = builder.nOut;
        scaleFactor = builder.scaleFactor;
        dropoutProbability = builder.dropoutProbability;
        scoreMode = builder.scoreMode;
        useCausalMask = builder.useCausalMask;
        withWeights = builder.withWeights;
        training = builder.training;

    }

    public DotProductAttentionLayer() {
    }


    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, RNNFormat.NCW,getLayerName());
    }


    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for Self Attention layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                    + inputType);
        }

        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;

        return InputType.recurrent(nIn, itr.getTimeSeriesLength());

    }

    @Override
    public void defineParameters(SDLayerParams params) {

    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {

    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable, SDVariable mask) {
        val q = paramTable.get(Q_KEY);
        val k = paramTable.get(K_KEY);
        val v = paramTable.get(V_KEY);
        val qMask = paramTable.get(Q_MASK_KEY);
        val vMask = paramTable.get(V_MASK_KEY);
        val kMask = paramTable.get(K_MASK_KEY);
        return sameDiff.nn.dotProductAttentionV2(q,v,k,qMask,vMask,scaleFactor,dropoutProbability,useCausalMask,training);

    }


    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<Builder> {

        private double scaleFactor;
        private double dropoutProbability;
        private int scoreMode;
        private boolean useCausalMask;
        private boolean withWeights;
        private boolean training;
        private long nIn;
        private long nOut;
        /**
         * @param scaleFactor Whether to scale the input or not.
         *               Defaults to true.
         */
        public Builder scale(double scaleFactor) {
            this.scaleFactor = scaleFactor;
            return this;
        }

        /**
         * @param nIn Number of inputs to the layer (input size)
         */
        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        /**
         * @param nOut Number of outputs (output size)
         */
        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }



        @Override
        @SuppressWarnings("unchecked")
        public DotProductAttentionLayer build() {
            return new DotProductAttentionLayer(this);
        }
    }
}
