/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.graph;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.convolution.KernelValidationUtil;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/** * LayerVertex is a GraphVertex with a neural network Layer (and, optionally an {@link InputPreProcessor}) in it
 * @author Alex Black
 */
@NoArgsConstructor
@Data
public class LayerVertex extends GraphVertex {

    private NeuralNetConfiguration layerConf;
    private InputPreProcessor preProcessor;
    //Set outputVertex to true when Layer is an OutputLayer, OR For use in specialized situations like reinforcement learning
    // For RL situations, this Layer insn't an OutputLayer, but is the last layer in a graph, that gets its error/epsilon
    // passed in externally
    private boolean outputVertex;


    public LayerVertex(NeuralNetConfiguration layerConf, InputPreProcessor preProcessor) {
        this.layerConf = layerConf;
        this.preProcessor = preProcessor;
    }

    @Override
    public GraphVertex clone() {
        return new LayerVertex(layerConf.clone(), (preProcessor != null ? preProcessor.clone() : null));
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof LayerVertex)) return false;
        LayerVertex lv = (LayerVertex) o;
        if (!layerConf.equals(lv.layerConf)) return false;
        if (preProcessor == null && lv.preProcessor != null || preProcessor != null && lv.preProcessor == null)
            return false;
        return preProcessor == null || preProcessor.equals(lv.preProcessor);
    }

    @Override
    public int hashCode() {
        return layerConf.hashCode() ^ (preProcessor != null ? preProcessor.hashCode() : 0);
    }

    @Override
    public int numParams(boolean backprop){
        return LayerFactories.getFactory(layerConf).initializer().numParams(layerConf,backprop);
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                      INDArray paramsView, boolean initializeParams) {
        //Now, we need to work out if this vertex is an output vertex or not...
        boolean isOutput = graph.getConfiguration().getNetworkOutputs().contains(name);

        return new org.deeplearning4j.nn.graph.vertex.impl.LayerVertex(
                graph, name, idx,
                LayerFactories.getFactory(layerConf).create(layerConf, null, idx, paramsView, initializeParams),
                preProcessor, isOutput);
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1) {
            throw new InvalidInputTypeException("LayerVertex expects exactly one input. Got: " + Arrays.toString(vertexInputs));
        }

        //Assume any necessary preprocessors have already been added
        Layer layer = layerConf.getLayer();
        if (layer instanceof ConvolutionLayer || layer instanceof SubsamplingLayer) {
            InputType.InputTypeConvolutional afterPreProcessor;
            if (preProcessor != null) {
                if (preProcessor instanceof FeedForwardToCnnPreProcessor) {
                    FeedForwardToCnnPreProcessor ffcnn = (FeedForwardToCnnPreProcessor) preProcessor;
                    afterPreProcessor = (InputType.InputTypeConvolutional) InputType.convolutional(ffcnn.getInputHeight(), ffcnn.getInputWidth(), ffcnn.getNumChannels());
                } else if (preProcessor instanceof RnnToCnnPreProcessor) {
                    RnnToCnnPreProcessor rnncnn = (RnnToCnnPreProcessor) preProcessor;
                    afterPreProcessor = (InputType.InputTypeConvolutional) InputType.convolutional(rnncnn.getInputHeight(), rnncnn.getInputWidth(), rnncnn.getNumChannels());
                } else {
                    //Assume no change to type of input...
                    //TODO checks for non convolutional input...
                    afterPreProcessor = (InputType.InputTypeConvolutional) vertexInputs[0];
                }
            } else {
                afterPreProcessor = (InputType.InputTypeConvolutional) vertexInputs[0];
            }

            int channelsOut;
            int[] kernel;
            int[] stride;
            int[] padding;
            if (layer instanceof ConvolutionLayer) {
                channelsOut = ((ConvolutionLayer) layer).getNOut();
                kernel = ((ConvolutionLayer) layer).getKernelSize();
                stride = ((ConvolutionLayer) layer).getStride();
                padding = ((ConvolutionLayer) layer).getPadding();
            } else {
                channelsOut = afterPreProcessor.getDepth();
                kernel = ((SubsamplingLayer) layer).getKernelSize();
                stride = ((SubsamplingLayer) layer).getStride();
                padding = ((SubsamplingLayer) layer).getPadding();
            }

            //First: check that the kernel size/stride/padding is valid
            int inHeight = afterPreProcessor.getHeight();
            int inWidth = afterPreProcessor.getWidth();
            KernelValidationUtil.validateShapes(inHeight, inWidth,
                    kernel[0], kernel[1], stride[0], stride[1],padding[0], padding[1]);

            int outWidth = (inWidth - kernel[1] + 2 * padding[1]) / stride[1] + 1;
            int outHeight = (inHeight - kernel[0] + 2 * padding[0]) / stride[0] + 1;

            return InputType.convolutional(outHeight,outWidth,channelsOut);
        } else if (layer instanceof BaseRecurrentLayer) {
            return InputType.recurrent(((BaseRecurrentLayer) layer).getNOut());
        } else if (layer instanceof FeedForwardLayer) {
            //Dense, autoencoder, etc
            return InputType.feedForward(((FeedForwardLayer) layer).getNOut());
        } else {
            //Unknown... probably same as input??
            return vertexInputs[0];
        }
    }
}
