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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;

import java.util.Collection;
import java.util.Map;

/**
 * 3D Convolutional Neural Network Loss Layer.<br> Handles calculation of gradients etc for various loss (objective)
 * functions.<br> NOTE: Cnn3DLossLayer does not have any parameters. Consequently, the output activations size is equal
 * to the input size.<br> Input and output activations are same as 3D CNN layers: 5 dimensions with one of two possible
 * shape, depending on the data format:<br> NCDHW ("channels first") format: data has shape
 * [miniBatchSize,channels,depth,height,width]<br> NDHWC ("channels last") format: data has shape
 * [miniBatchSize,channels,depth,height,width]<br> Cnn3DLossLayer has support for a built-in activation function (tanh,
 * softmax etc) - if this is not required, set activation function to Activation.IDENTITY. For activations such as
 * softmax, note that this is applied channel-wise: that is, softmax is applied along dimension 1 for NCDHW, or
 * dimension 4 for NDHWC for each minibatch, and x/y/z location separately.<br>
 * <br>
 * Note that multiple types of masking are supported. Mask arrays (when present) must be 5d in a 'broadcastable' format:
 * that is, for (n=minibatchSize, c=channels, d=depth, h=height, w=width):<br> - Per example masking: Where an example
 * is present or not (and all outputs are masked by it). Mask shape [n,1,1,1,1] for both NCDHW and NDHWC<br> - Per x/y/z
 * location masking: where each spatial X/Y/Z location is present or not (all channels at a given x/y/z are masked by
 * it). Mask shape: [n,1,d,h,w] (NCDHW format) or [n,d,h,w,1] (NDHWC format).<br> - Per output masking: Where each
 * output activation value is present or not - mask shape [n,c,d,h,w] (NCDHW format) or [n,d,h,w,c] (NDHWC format) -
 * same as input/output in both cases<br>
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Cnn3DLossLayer extends FeedForwardLayer {

    protected ILossFunction lossFn;
    protected Convolution3D.DataFormat dataFormat;

    private Cnn3DLossLayer(Builder builder) {
        super(builder);
        this.lossFn = builder.lossFn;
        this.dataFormat = builder.dataFormat;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        org.deeplearning4j.nn.layers.convolution.Cnn3DLossLayer ret =
                        new org.deeplearning4j.nn.layers.convolution.Cnn3DLossLayer(conf, networkDataType);
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
        if (inputType == null || (inputType.getType() != InputType.Type.CNN3D
                        && inputType.getType() != InputType.Type.CNNFlat)) {
            throw new IllegalStateException("Invalid input type for CnnLossLayer (layer index = " + layerIndex
                            + ", layer name=\"" + getLayerName() + "\"): Expected CNN3D or CNNFlat input, got "
                            + inputType);
        }
        return inputType;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreProcessorForInputTypeCnn3DLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //During inference and training: dup the input array. But, this counts as *activations* not working memory
        return new LayerMemoryReport.Builder(layerName, getClass(), inputType, inputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }


    @Getter
    @Setter
    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        /**
         * Format of the input/output data. See {@link Convolution3D.DataFormat} for details
         */
        protected Convolution3D.DataFormat dataFormat;

        /**
         * @param format Format of the input/output data. See {@link Convolution3D.DataFormat} for details
         */
        public Builder(@NonNull Convolution3D.DataFormat format) {
            this.setDataFormat(format);
            this.setActivationFn(Activation.IDENTITY.getActivationFunction());
        }

        @Override
        @SuppressWarnings("unchecked")
        public Builder nIn(int nIn) {
            throw new UnsupportedOperationException(
                            "Cnn3DLossLayer has no parameters, thus nIn will always equal nOut.");
        }

        @Override
        @SuppressWarnings("unchecked")
        public Builder nOut(int nOut) {
            throw new UnsupportedOperationException(
                            "Cnn3DLossLayer has no parameters, thus nIn will always equal nOut.");
        }

        @Override
        public void setNIn(long nIn){
            throw new UnsupportedOperationException(
                    "Cnn3DLossLayer has no parameters, thus nIn will always equal nOut.");
        }

        @Override
        public void setNOut(long nOut){
            throw new UnsupportedOperationException(
                    "Cnn3DLossLayer has no parameters, thus nIn will always equal nOut.");
        }


        @Override
        @SuppressWarnings("unchecked")
        public Cnn3DLossLayer build() {
            return new Cnn3DLossLayer(this);
        }
    }
}
