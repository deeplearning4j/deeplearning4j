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

package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Collection;
import java.util.Map;

/**
 * Convolutional Neural Network Loss Layer.<br>
 * Handles calculation of gradients etc for various loss (objective) functions.<br>
 * NOTE: CnnLossLayer does not have any parameters. Consequently, the output activations size is equal to the input size.<br>
 * Input and output activations are same as other CNN layers: 4 dimensions with shape [miniBatchSize,channels,height,width]<br>
 * CnnLossLayer has support for a built-in activation function (tanh, softmax etc) - if this is not required, set
 * activation function to Activation.IDENTITY. For activations such as softmax, note that this is applied depth-wise:
 * that is, softmax is applied along dimension 1 (depth) for each minibatch, and x/y location separately.<br>
 * <br>
 * Note that 3 types of masking are supported, where (n=minibatchSize, c=channels, h=height, w=width):<br>
 * - Per example masking: Where an example is present or not (and all outputs are masked by it). Mask shape [n,1]<br>
 * - Per x/y location masking: where each spatial X/Y location is present or not (all channels at a given x/y are masked by it).
 * Mask shape: [n,h,w].<br>
 * - Per output masking: Where each output activation value is present or not - mask shape [n,c,h,w] (same as output)<br>
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class CnnLossLayer extends FeedForwardLayer {

    protected ILossFunction lossFn;

    private CnnLossLayer(Builder builder) {
        super(builder);
        this.lossFn = builder.lossFn;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.CnnLossLayer ret =
                        new org.deeplearning4j.nn.layers.convolution.CnnLossLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || (inputType.getType() != InputType.Type.CNN && inputType.getType() != InputType.Type.CNNFlat)) {
            throw new IllegalStateException("Invalid input type for CnnLossLayer (layer index = " + layerIndex
                            + ", layer name=\"" + getLayerName() + "\"): Expected CNN or CNNFlat input, got " + inputType);
        }
        return inputType;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //During inference and training: dup the input array. But, this counts as *activations* not working memory
        return new LayerMemoryReport.Builder(layerName, LossLayer.class, inputType, inputType).standardMemory(0, 0) //No params
                .workingMemory(0, 0, 0, 0)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }


    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        public Builder() {
            this.activationFn = Activation.IDENTITY.getActivationFunction();
        }

        public Builder(LossFunction lossFunction) {
            lossFunction(lossFunction);
        }

        public Builder(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
        }

        @Override
        @SuppressWarnings("unchecked")
        public Builder nIn(int nIn) {
            throw new UnsupportedOperationException("Ths layer has no parameters, thus nIn will always equal nOut.");
        }

        @Override
        @SuppressWarnings("unchecked")
        public Builder nOut(int nOut) {
            throw new UnsupportedOperationException("Ths layer has no parameters, thus nIn will always equal nOut.");
        }

        @Override
        @SuppressWarnings("unchecked")
        public CnnLossLayer build() {
            return new CnnLossLayer(this);
        }
    }
}
